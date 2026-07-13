import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor
from app.models.base_model import BaseOCRModel
from app.utils.logger import setup_logger
from app.utils.constants import TASK_PROMPTS

logger = setup_logger("paddleocr_vl_model")

class PaddleOCRVLModel(BaseOCRModel):
    def load(self) -> None:
        logger.info(f"Loading PaddleOCR-VL model from: {self.model_path}")
        
        # Determine device
        device_type = self.device
        if device_type == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA is not available, falling back to CPU.")
            device_type = "cpu"
            
        if device_type == "cpu":
            torch_dtype = torch.float32
        else:
            torch_dtype = torch.bfloat16
        
        # Register standard/default RoPE initialization function in transformers if missing (transformers 5.x compatibility)
        from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS
        if "default" not in ROPE_INIT_FUNCTIONS:
            logger.info("Registering 'default' rope initialization function in transformers.")
            def _compute_default_scaling_rope_parameters(config, device=None, seq_len=None, layer_type=None):
                config.standardize_rope_params()
                rope_parameters_dict = config.rope_parameters[layer_type] if layer_type is not None else config.rope_parameters
                base = rope_parameters_dict["rope_theta"]
                partial_rotary_factor = rope_parameters_dict.get("partial_rotary_factor", 1.0)
                head_dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
                dim = int(head_dim * partial_rotary_factor)
                attention_factor = 1.0
                inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.int64).to(device=device, dtype=torch.float) / dim))
                return inv_freq, attention_factor
            
            ROPE_INIT_FUNCTIONS["default"] = _compute_default_scaling_rope_parameters

        # Load model and processor
        # Set trust_remote_code=True since PaddleOCR-VL uses custom modeling code
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            trust_remote_code=True,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True
        )
        if device_type != "cpu":
            self.model = self.model.to(device_type)
            
        self.model = self.model.eval()
        
        self.processor = AutoProcessor.from_pretrained(
            self.model_path,
            trust_remote_code=True
        )
        self.device = device_type
        logger.info("PaddleOCR-VL model and processor loaded successfully.")

    def predict(self, image: Image.Image, task: str, max_new_tokens: int = 1024) -> str:
        if self.model is None or self.processor is None:
            self.load()
            
        prompt = TASK_PROMPTS.get(task, "OCR:")
        
        # PaddleOCR-VL chat format
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt}
                ]
            }
        ]
        
        # Apply processor's chat template
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt"
        ).to(self.device)
        
        # Inference
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False
            )
            
        # Decode and extract response (slice to remove the prompt prefix)
        input_len = inputs.input_ids.shape[1]
        generated_tokens = outputs[:, input_len:]
        decoded = self.processor.batch_decode(generated_tokens, skip_special_tokens=True)[0]
        return decoded
