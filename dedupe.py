import argparse
import utils

def main(args):
    utils.setup_logging()
    try:
        lines = utils.read_file(args.input)
        deduped = utils.dedupe(lines, model_name=args.model, threshold=args.threshold)
        utils.write_file(args.output, deduped)
    except Exception as e:
        logging.error(f"Error during deduplication process: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Deduplicate text file based on embeddings.")
    parser.add_argument("--input", type=str, required=True, help="Input file path containing text lines.")
    parser.add_argument("--output", type=str, required=True, help="Output file path for deduplicated lines.")
    parser.add_argument("--model", type=str, default="Snowflake/snowflake-arctic-embed-xs", help="Embedding model to use.")
    parser.add_argument("--threshold", type=float, default=0.97, help="Similarity threshold for deduplication.")

    args = parser.parse_args()
    main(args)
