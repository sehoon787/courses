- https://github.com/webdataset/webdataset
- https://pytorch.org/data/0.7/generated/torchdata.datapipes.iter.WebDataset.html

- ```
  
  
  import os
import tarfile
from pathlib import Path
import argparse

def create_webdataset_from_flac(input_dir, output_tar, suffix=".flac"):
    input_dir = Path(input_dir)
    output_tar = Path(output_tar)

    with tarfile.open(output_tar, "w") as tar:
        for idx, flac_file in enumerate(sorted(input_dir.glob(f"*{suffix}"))):
            key = f"{idx:06d}"  # 예: 000001, 000002 처럼
            arcname = f"{key}.flac"

            tar.add(flac_file, arcname=arcname)

            # 만약 각 flac 파일에 대응하는 transcript 텍스트 파일(.txt)이 있으면 같이 넣기
            txt_file = flac_file.with_suffix('.txt')
            if txt_file.exists():
                tar.add(txt_file, arcname=f"{key}.txt")

    print(f"Created WebDataset tar: {output_tar}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pack FLAC files into a WebDataset tar.")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory with .flac files (and optional .txt)")
    parser.add_argument("--output_tar", type=str, required=True, help="Path to output .tar file")
    args = parser.parse_args()

    create_webdataset_from_flac(args.input_dir, args.output_tar)
```
