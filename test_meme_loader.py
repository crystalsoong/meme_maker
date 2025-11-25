from utils.data_utils import build_hf_meme_manifest, verify_manifest

if __name__ == "__main__":
    path = build_hf_meme_manifest()
    print("Saved manifest:", path)
    verify_manifest(path, n=5)
