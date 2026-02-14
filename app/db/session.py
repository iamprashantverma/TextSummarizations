from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from app.core.config import settings

#create_engine is used to establish a connection 
#configuration between the application and the database.
engine = create_engine(
    settings.DATABASE_URL,
    pool_pre_ping=True #Checks the database connection
    #before using it and automatically reconnects if the connection is stale.
)

SessionLocal = sessionmaker(
    autocommit=False, #ensures transactions are committed manually, giving better control and preventing accidental database writes
    autoflush=False, #Prevents SQLAlchemy from automatically sending changes to the database before queries are executed
    bind=engine # Connects the session factory to the database engine so all sessions know which database to use.

)
