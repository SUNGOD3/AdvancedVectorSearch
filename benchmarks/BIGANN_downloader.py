from pathlib import Path
from tqdm import tqdm
import logging
import tarfile
import ftplib

class SIFTDatasetDownloader:
    def __init__(self, config: dict):
        self.config = config
        self.logger = self._setup_logger()
        
    def _setup_logger(self):
        logger = logging.getLogger('SIFTDatasetDownloader')
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        return logger
    
    def download_via_ftp(self, filename: str):
        local_path = Path(self.config['local_path']) / filename
        local_path.parent.mkdir(parents=True, exist_ok=True)
        ftp_host = 'ftp.irisa.fr'
        ftp_path = '/local/texmex/corpus'
        
        self.logger.info(f"Connecting to FTP server {ftp_host}")
        
        try:
            # 連接FTP服務器
            ftp = ftplib.FTP(ftp_host)
            ftp.login()  # 匿名登錄
            ftp.cwd(ftp_path)
            
            file_size = ftp.size(filename)
            
            with open(local_path, 'wb') as f:
                with tqdm(total=file_size, unit='B', unit_scale=True) as pbar:
                    def callback(data):
                        f.write(data)
                        pbar.update(len(data))
                    
                    ftp.retrbinary(f'RETR {filename}', callback)
            
            ftp.quit()
            return local_path
            
        except Exception as e:
            self.logger.error(f"Error downloading via FTP: {str(e)}")
            raise
    
    def extract_tar_gz(self, tar_path: Path):
        extract_path = tar_path.parent
        self.logger.info(f"Extracting {tar_path} to {extract_path}")
        
        with tarfile.open(tar_path, 'r:gz') as tar:
            tar.extractall(path=extract_path)
    
    def setup_dataset(self):
        try:
            filename = 'siftsmall.tar.gz'
            tar_path = self.download_via_ftp(filename)
            
            self.extract_tar_gz(tar_path)
            
            self.logger.info("Dataset setup completed successfully!")
            
        except Exception as e:
            self.logger.error(f"Error setting up dataset: {str(e)}")
            raise


config = {
    'dataset_name': 'ANN_SIFT10K',
    'base_url': 'ftp://ftp.irisa.fr/local/texmex/corpus',
    'local_path': './data/sift10k'
}

downloader = SIFTDatasetDownloader(config)
downloader.setup_dataset()