import sys
import time
import hydra
from omegaconf import DictConfig
from loguru import logger

from src.measurements.scraper_service import ScraperService

# Check if loop argument is passed and clean it up before Hydra parses sys.argv
is_loop = False
if "--loop" in sys.argv:
    is_loop = True
    sys.argv.remove("--loop")

@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    logger.info("Initializing Scraper Service...")
    scraper = ScraperService(cfg)
    
    regions = list(cfg.scraper.regions.keys())
    logger.info(f"Configured regions to scrape: {regions}")
    
    if is_loop:
        interval_min = cfg.scraper.interval_minutes
        logger.info(f"Running in LOOP mode. Scrapes will run every {interval_min} minutes. Press Ctrl+C to exit.")
        try:
            while True:
                start_time = time.time()
                logger.info("Executing periodic scrape cycle...")
                for region in regions:
                    try:
                        scraper.scrape_region(region)
                    except Exception as e:
                        logger.error(f"Error scraping region {region}: {e}")
                
                elapsed = time.time() - start_time
                sleep_seconds = max(0.0, (interval_min * 60) - elapsed)
                logger.info(f"Scrape cycle finished in {elapsed:.2f}s. Sleeping for {sleep_seconds/60:.2f} minutes...")
                time.sleep(sleep_seconds)
        except KeyboardInterrupt:
            logger.info("Loop mode stopped by user.")
    else:
        logger.info("Running in SINGLE-RUN mode.")
        success = True
        for region in regions:
            try:
                res = scraper.scrape_region(region)
                if not res:
                    success = False
            except Exception as e:
                logger.error(f"Error scraping region {region}: {e}")
                success = False
        
        if success:
            logger.info("Single scrape run completed successfully.")
        else:
            logger.warning("Single scrape run completed with some errors.")

if __name__ == "__main__":
    main()
