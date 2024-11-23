from .FasterLivePortrait_node import LivePortraitModelLoader, FasterLivePortraitProcess


NODE_CLASS_MAPPINGS = {
    "LivePortraitModelLoader": LivePortraitModelLoader,
    "FasterLivePortraitProcess": FasterLivePortraitProcess,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LivePortraitModelLoader": "LivePortraitModelLoader",
    "FasterLivePortraitProcess": "FasterLivePortraitProcess",
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']