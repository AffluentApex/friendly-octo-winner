import os
import urllib.request
import gzip
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def download_and_extract():
    """Download MNIST dataset from the official source and extract it"""
    try:
        # Create data directory
        data_dir = 'data'
        os.makedirs(data_dir, exist_ok=True)
        
        # MNIST dataset URLs
        base_url = 'http://yann.lecun.com/exdb/mnist/'
        files = {
            'train-images-idx3-ubyte.gz': 'training set images',
            'train-labels-idx1-ubyte.gz': 'training set labels',
            't10k-images-idx3-ubyte.gz': 'test set images',
            't10k-labels-idx1-ubyte.gz': 'test set labels'
        }
        
        # Download and extract each file
        for filename, description in files.items():
            # Download
            url = base_url + filename
            output_gz = os.path.join(data_dir, filename)
            output_file = output_gz[:-3]  # Remove .gz extension
            
            if not os.path.exists(output_file):
                logger.info(f"Downloading {description} ({filename})...")
                try:
                    urllib.request.urlretrieve(url, output_gz)
                except Exception as e:
                    # If primary URL fails, try backup URL
                    backup_url = f'https://ossci-datasets.s3.amazonaws.com/mnist/{filename}'
                    logger.info(f"Primary download failed, trying backup URL...")
                    urllib.request.urlretrieve(backup_url, output_gz)
                
                # Extract
                logger.info(f"Extracting {filename}...")
                with gzip.open(output_gz, 'rb') as f_in:
                    with open(output_file, 'wb') as f_out:
                        f_out.write(f_in.read())
                
                # Remove gz file
                os.remove(output_gz)
                logger.info(f"Extracted {filename}")
            else:
                logger.info(f"File {output_file} already exists, skipping...")
        
        logger.info("\nDataset download and extraction completed successfully!")
        logger.info(f"Files are located in: {os.path.abspath(data_dir)}")
        return True
        
    except Exception as e:
        logger.error(f"Error downloading/extracting dataset: {e}")
        return False

if __name__ == '__main__':
    download_and_extract()
