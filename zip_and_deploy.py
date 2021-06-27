from pathlib import Path
import shutil
import zipfile

assets_path = Path('yolov5-lambda')
dirs_to_remove = [assets_path / '__pycache__', assets_path / 'utils' / '__pycache__',
                    assets_path / 'models' / '__pycache__', 
                    assets_path / 'runs']

def remove_dirs(dirs_to_remove): 
    for dir_to_remove in dirs_to_remove: 
        try: 
            shutil.rmtree(dir_to_remove)
        except:
            pass

def walk_folder(path):
    """
    Recursively iterates in a folder - similar to os.walk. 
    """
    for item_path in path.iterdir():
        yield item_path
        if item_path.is_dir():
            yield from walk_folder(item_path)
            continue

def zip_folder(assets_path):
    zf = zipfile.ZipFile(f"{assets_path.name}.zip", "w", zipfile.ZIP_DEFLATED)
    for item_path in walk_folder(assets_path):
        relative_path = item_path.relative_to(assets_path)
        print('zipping', relative_path)
        zf.write(item_path, relative_path)
    zf.close()


if __name__ == '__main__': 
    print('running zip_and_deploy.py')
    remove_dirs(dirs_to_remove)
    zip_folder(assets_path)

