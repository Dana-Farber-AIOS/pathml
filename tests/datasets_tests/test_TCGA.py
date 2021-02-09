import sys
sys.path.append('/mnt/disks/data_disk/pathml/pathml/datasets/')
import TCGA

def test_TCGA(tissue_codes, outdir = '.'):
  print(tissue_codes, outdir)
  print("warning, will download significant datafiles to", outdir)
  test_TCGA_obj = TCGA.TCGA(tissue_codes = tissue_codes, outdir = outdir)

if __name__ == "__main__":
  test_TCGA(tissue_codes = ['TCGA-LUSC'], outdir = '/mnt/disks/data_disk/LUSC')