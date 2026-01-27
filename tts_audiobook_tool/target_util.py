import os
from pathlib import Path


class TargetUtil:
    """
    "Target" is app nomenclature for:
    1) a huggingface repo id or 
    2) a local directory path to a model presumably downloaded from a huggingface repo
    """

    @staticmethod
    def is_hf_repo_id_syntax_valid(s: str) -> bool:
        
        from huggingface_hub.utils._validators import validate_repo_id
        from huggingface_hub.errors import HFValidationError

        try:
            validate_repo_id(s)
        except HFValidationError:
            return False
        return True

    @staticmethod
    def exist_test(
            target: str,
            local_file_markers = ["config.json", "model.safetensors", "adapter_config.json"]
    ) -> tuple[str, str]:
        """
        Tests if hf target exists

        Returns 
            Tuple with mutually exclusive values:
                source type (local_path | hf_cache | hf_remote)
                error string 
        """

        from huggingface_hub import model_info, try_to_load_from_cache
        from huggingface_hub.errors import RepositoryNotFoundError, HFValidationError

        # If value is an existing local directory, check for existence a 'local file marker'        
        if os.path.exists(target):
            if not os.path.isdir(target):
                return "", "Path is not a directory"
            is_hf_dir = any(os.path.isfile(os.path.join(target, marker)) for marker in local_file_markers)
            if is_hf_dir:
                return "local_path", ""
            else:
                return "", "Directory is missing typical hf file markers"
            
        error = "No such local directory;\n"

        # Test for correct hf repo id string format
        if not TargetUtil.is_hf_repo_id_syntax_valid(target):
            error += "Invalid string format for hf repo id"
            return "", error

        # Check local hf cache
        for file_marker in local_file_markers:
            cached_path = try_to_load_from_cache(target, file_marker)
            if isinstance(cached_path, str):
                return "hf_cache", ""
        
        # Test for hf repo remote reachability
        try:
            model_info(target)
            return "hf_remote", ""
        except RepositoryNotFoundError:
            error += "Repo not found or gated"
            return "", error
        except Exception as e:
            error += f"Network or other error: {e}"
            return "", error

    @staticmethod
    def is_same_target(a: str, b: str) -> bool:
        
        if TargetUtil.is_hf_repo_id_syntax_valid(b):
            is_same = (a == b)
        else:
            # Treat as dir path
            try:
                is_same = Path(b).exists() and Path(b).samefile(Path(a))
            except:
                is_same = False
        return is_same


