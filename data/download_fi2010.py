from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    target = root / "data" / "fi2010"
    target.mkdir(parents=True, exist_ok=True)

    print("FI-2010 dataset setup")
    print("----------------------")
    print("1) Download FI-2010 from the official source:")
    print("   https://etsin.fairdata.fi/dataset/73eb48d7-4dbc-4a10-a52a-da745b47a649")
    print("2) Extract the archive.")
    print("3) Copy the .txt files into:")
    print(f"   {target}")
    print("")
    print("Expected files example:")
    print("  Train_Dst_NoAuction_ZScore_CF_1.txt")
    print("  Test_Dst_NoAuction_ZScore_CF_1.txt")
    print("")
    print("When done, update configs/lit_like.yaml with the correct paths.")


if __name__ == "__main__":
    main()
