import torch
import numpy as np
import gc
from tqdm import tqdm
from chronos import ChronosPipeline
from peft import LoraConfig, get_peft_model, TaskType
from utils import get_device, clear_gpu_memory


class ChronosLoRAModel:
    """
    Amazon Chronos-T5-Small with optional LoRA fine-tuning via HF PEFT.
    Supports zero-shot inference and LoRA-adapted fine-tuning.
    """

    def __init__(self, config: dict):
        self.config = config
        self.model_id = config["model"]["chronos_id"]
        self.device = get_device(config)
        self.pipeline = None
        self.peft_model = None
        self.lora_applied = False

    def load_pipeline(self):
        """Load the Chronos pipeline with GPU if available."""
        from utils import print_device_info
        print_device_info(self.config)

        device_map = "cuda" if self.device.type == "cuda" else "cpu"
        # Use bfloat16 on GPU for memory efficiency, float32 on CPU
        dtype = torch.bfloat16 if self.device.type == "cuda" else torch.float32

        self.pipeline = ChronosPipeline.from_pretrained(
            self.model_id,
            device_map=device_map,
            dtype=dtype,
        )
        return self

    def apply_lora(self):
        """
        Extract the T5 model from the pipeline, enable gradient checkpointing,
        then apply LoRA via PEFT.
        """
        if self.pipeline is None:
            raise ValueError("Pipeline not loaded. Call load_pipeline() first.")

        model = self.pipeline.model.model  # T5ForConditionalGeneration

        # Enable gradient checkpointing BEFORE applying LoRA
        model.gradient_checkpointing_enable()

        lora_config = LoraConfig(
            r=self.config["model"]["lora_rank"],
            lora_alpha=self.config["model"]["lora_alpha"],
            lora_dropout=self.config["model"]["lora_dropout"],
            target_modules=self.config["model"]["lora_target_modules"],
            task_type=TaskType.SEQ_2_SEQ_LM,
            bias="none",
        )

        self.peft_model = get_peft_model(model, lora_config)
        self.peft_model.print_trainable_parameters()
        self.lora_applied = True

        return self

    def predict_zero_shot(self, context: np.ndarray, horizon: int = 48) -> np.ndarray:
        """
        Zero-shot prediction using Chronos pipeline.
        Returns median/point forecast.
        """
        if self.pipeline is None:
            raise ValueError("Pipeline not loaded. Call load_pipeline() first.")

        context_tensor = torch.tensor(context, dtype=torch.float32)

        with torch.no_grad():
            forecast = self.pipeline.predict(
                inputs=context_tensor,
                prediction_length=horizon,
                limit_prediction_length=False,
            )

        # forecast shape: (num_samples, horizon) -- take median
        median_forecast = torch.median(forecast, dim=0).values.cpu().numpy()
        return median_forecast

    def train_lora(self, train_contexts: list, train_targets: list,
                   epochs: int = 10, dry_run: bool = False, config: dict = None):
        """
        Fine-tune the LoRA-adapted model on training data.
        Implements mixed precision, gradient accumulation, memory management, and validation loss tracking.
        
        Returns:
            dict: {'train_loss': list, 'val_loss': list} - loss history for each epoch
        """
        if not self.lora_applied:
            raise ValueError("LoRA not applied. Call apply_lora() first.")

        cfg = config or self.config
        per_device_batch_size = cfg["training"]["per_device_batch_size"]
        gradient_accumulation_steps = cfg["training"]["gradient_accumulation_steps"]
        dry_run_cfg = cfg.get("dry_run", {})
        max_steps = dry_run_cfg.get("max_steps", 5) if dry_run else None
        max_epochs = dry_run_cfg.get("max_epochs", 1) if dry_run else epochs

        # Split into train/val (80/20)
        split_idx = int(len(train_contexts) * 0.8)
        val_contexts = train_contexts[split_idx:]
        val_targets = train_targets[split_idx:]
        train_contexts = train_contexts[:split_idx]
        train_targets = train_targets[:split_idx]

        model = self.peft_model
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=1e-4,
            weight_decay=0.01,
        )

        scaler = torch.amp.GradScaler("cuda") if self.device.type == "cuda" else None
        use_amp = self.device.type == "cuda"

        tokenizer = self.pipeline.tokenizer
        native_pred_len = tokenizer.config.prediction_length  # 64 for chronos-t5-small
        
        # Track loss history
        history = {'train_loss': [], 'val_loss': []}
        total_steps = 0

        for epoch in range(max_epochs):
            # ============ TRAINING ============
            model.train()
            epoch_loss = 0.0
            optimizer.zero_grad()
            step_in_epoch = 0
            n_batches = (len(train_contexts) + per_device_batch_size - 1) // per_device_batch_size

            pbar = tqdm(
                range(0, len(train_contexts), per_device_batch_size),
                desc=f"  Epoch {epoch+1}/{max_epochs} [Train]",
                unit="batch",
                total=n_batches,
                leave=True,
            )
            for i in pbar:
                batch_contexts = train_contexts[i:i + per_device_batch_size]
                batch_targets = train_targets[i:i + per_device_batch_size]

                for ctx, tgt in zip(batch_contexts, batch_targets):
                    ctx_tensor = torch.tensor(ctx, dtype=torch.float32).unsqueeze(0)
                    tgt_tensor = torch.tensor(tgt, dtype=torch.float32).unsqueeze(0)

                    # Pad target to native prediction_length if shorter
                    if tgt_tensor.shape[1] < native_pred_len:
                        pad_len = native_pred_len - tgt_tensor.shape[1]
                        tgt_tensor = torch.nn.functional.pad(tgt_tensor, (0, pad_len), value=0.0)

                    # Tokenize on CPU (tokenizer internals are on CPU), then move to device
                    input_ids, attention_mask, scale = tokenizer.context_input_transform(
                        ctx_tensor
                    )
                    labels, labels_mask = tokenizer.label_input_transform(
                        tgt_tensor, scale
                    )
                    input_ids = input_ids.to(self.device)
                    attention_mask = attention_mask.to(self.device)
                    labels = labels.to(self.device)

                    if use_amp:
                        with torch.amp.autocast("cuda"):
                            output = model(
                                input_ids=input_ids,
                                attention_mask=attention_mask,
                                labels=labels,
                            )
                            loss = output.loss / gradient_accumulation_steps
                        scaler.scale(loss).backward()
                    else:
                        output = model(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            labels=labels,
                        )
                        loss = output.loss / gradient_accumulation_steps
                        loss.backward()

                    epoch_loss += loss.item() * gradient_accumulation_steps
                    pbar.set_postfix(loss=f"{epoch_loss / max(step_in_epoch + 1, 1):.4f}")

                step_in_epoch += 1

                if step_in_epoch % gradient_accumulation_steps == 0:
                    if use_amp:
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()
                    optimizer.zero_grad()
                    total_steps += 1

                    if max_steps is not None and total_steps >= max_steps:
                        break

            pbar.close()

            # Handle remaining gradients
            if step_in_epoch % gradient_accumulation_steps != 0:
                if use_amp:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad()

            avg_train_loss = epoch_loss / max(step_in_epoch, 1)
            history['train_loss'].append(avg_train_loss)

            # ============ VALIDATION ============
            model.eval()
            val_epoch_loss = 0.0
            val_step = 0

            val_pbar = tqdm(
                range(0, len(val_contexts), per_device_batch_size),
                desc=f"  Epoch {epoch+1}/{max_epochs} [Val]",
                unit="batch",
                total=(len(val_contexts) + per_device_batch_size - 1) // per_device_batch_size,
                leave=False,
            )
            
            with torch.no_grad():
                for i in val_pbar:
                    batch_contexts = val_contexts[i:i + per_device_batch_size]
                    batch_targets = val_targets[i:i + per_device_batch_size]

                    for ctx, tgt in zip(batch_contexts, batch_targets):
                        ctx_tensor = torch.tensor(ctx, dtype=torch.float32).unsqueeze(0)
                        tgt_tensor = torch.tensor(tgt, dtype=torch.float32).unsqueeze(0)

                        # Pad target to native prediction_length if shorter
                        if tgt_tensor.shape[1] < native_pred_len:
                            pad_len = native_pred_len - tgt_tensor.shape[1]
                            tgt_tensor = torch.nn.functional.pad(tgt_tensor, (0, pad_len), value=0.0)

                        # Tokenize
                        input_ids, attention_mask, scale = tokenizer.context_input_transform(
                            ctx_tensor
                        )
                        labels, labels_mask = tokenizer.label_input_transform(
                            tgt_tensor, scale
                        )
                        input_ids = input_ids.to(self.device)
                        attention_mask = attention_mask.to(self.device)
                        labels = labels.to(self.device)

                        output = model(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            labels=labels,
                        )
                        val_epoch_loss += output.loss.item()

                    val_step += 1
                    val_pbar.update(1)

            val_pbar.close()
            avg_val_loss = val_epoch_loss / max(val_step, 1)
            history['val_loss'].append(avg_val_loss)

            print(f"  Epoch {epoch+1}/{max_epochs}, Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}")

            # Memory management
            clear_gpu_memory()

            if max_steps is not None and total_steps >= max_steps:
                print(f"  Dry-run: reached max_steps={max_steps}")
                break

        model.eval()
        return history

    def predict_lora(self, context: np.ndarray, horizon: int = 48) -> np.ndarray:
        """
        Predict using the LoRA-adapted model.
        Uses the underlying Chronos pipeline's predict method which uses the modified model.
        """
        if not self.lora_applied:
            raise ValueError("LoRA not applied.")

        # Temporarily set the PEFT model back
        self.peft_model.eval()

        context_tensor = torch.tensor(context, dtype=torch.float32)

        with torch.no_grad():
            forecast = self.pipeline.predict(
                inputs=context_tensor,
                prediction_length=horizon,
                limit_prediction_length=False,
            )

        median_forecast = torch.median(forecast, dim=0).values.cpu().numpy()
        return median_forecast


if __name__ == "__main__":
    from utils import load_config, set_global_seed

    config = load_config()
    set_global_seed(config["execution"]["random_seed"])

    chronos = ChronosLoRAModel(config)
    chronos.load_pipeline()
    print("Chronos pipeline loaded.")

    # Test zero-shot
    dummy_context = np.random.randn(336) * 1000 + 30000
    pred = chronos.predict_zero_shot(dummy_context, horizon=48)
    print(f"Zero-shot prediction shape: {pred.shape}, mean: {pred.mean():.1f}")
