from pathlib import Path
import shutil
import zipfile

assets_path = Path('lambda-yolov5-felixh')
dirs_to_remove = [assets_path / '__pycache__',
                    assets_path / 'runs']

for dir_to_remove in dirs_to_remove: 
    try: 
        shutil.rmtree(dir_to_remove)
    except:
        pass

def zip_folder(assets_path):
    zf = zipfile.ZipFile(f"{assets_path.name}.zip", "w", zipfile.ZIP_DEFLATED)
    for item_path in assets_path.iterdir():
        relative_path = item_path.relative_to(assets_path)
        print('zipping', relative_path)
        zf.write(item_path, relative_path)
    zf.close()

zip_folder(assets_path)

