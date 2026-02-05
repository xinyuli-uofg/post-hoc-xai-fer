

import datetime
from xai_core import add_common_cli, build_model, load_val_meta, ensure_dir, iter_images, run_core_outputs

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Core outputs only (no post-hoc XAI).")
    add_common_cli(parser)
    args = parser.parse_args()

    model, _ = build_model()
    path2meta = load_val_meta(data_dir=args.data_dir)

    imgs = iter_images(args.data_dir)
    print(f"\nProcessing {len(imgs)} images...\n")
    t0 = datetime.datetime.now()

    ensure_dir(args.out_root)
    for p in imgs:
        run_core_outputs(model, p, path2meta, args.out_root)

    print("Runtime:", datetime.datetime.now() - t0)

if __name__ == "__main__":
    main()
