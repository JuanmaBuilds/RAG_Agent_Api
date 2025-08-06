from pymongo import MongoClient
from pymongo.errors import ConnectionFailure
from typing import Optional
import logging

from .config import settings

logger = logging.getLogger(__name__)


class MongoDBManager:
    """MongoDB Atlas connection manager."""
    
    def __init__(self):
        self.client: Optional[MongoClient] = None
        self.database = None
        self.collection = None
    
    async def connect(self):
        """Establish connection to MongoDB Atlas."""
        try:
            # Add SSL parameters to handle TLS issues
            connection_string = settings.mongodb_atlas_uri
            if "?" not in connection_string:
                connection_string += "?ssl=true"
            elif "ssl=" not in connection_string:
                connection_string += "&ssl=true"
            
            self.client = MongoClient(connection_string, serverSelectionTimeoutMS=5000)
            # Test the connection
            self.client.admin.command('ping')
            
            self.database = self.client[settings.mongodb_database_name]
            self.collection = self.database[settings.mongodb_collection_name]
            
            logger.info("Successfully connected to MongoDB Atlas")
            
        except ConnectionFailure as e:
            logger.error(f"Failed to connect to MongoDB Atlas: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error connecting to MongoDB: {e}")
            raise
    
    async def disconnect(self):
        """Close MongoDB connection."""
        if self.client:
            self.client.close()
            logger.info("MongoDB connection closed")
    
    def get_collection(self):
        """Get the documents collection."""
        return self.collection
    
    def get_database(self):
        """Get the database instance."""
        return self.database


# Global database manager instance
db_manager = MongoDBManager() 