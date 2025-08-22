"""
Synthetic data generator for data governance queries.
Generates realistic but fake datasets for testing the RAG system.
"""

import json
import random
import pandas as pd
from typing import List, Dict, Any
from datetime import datetime, timedelta
import os
import sys

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import data_config

class SyntheticDataGenerator:
    """Generates synthetic data governance datasets."""
    
    def __init__(self):
        self.departments = [
            "Engineering", "Marketing", "Sales", "Finance", 
            "HR", "Operations", "Legal", "Product"
        ]
        
        self.data_types = [
            "customer", "transaction", "user", "product", "inventory",
            "financial", "analytics", "log", "audit", "metadata"
        ]
        
        self.table_prefixes = [
            "raw_", "processed_", "aggregated_", "staging_", "final_",
            "temp_", "archive_", "backup_", "replica_", "view_"
        ]
        
        self.column_types = [
            "id", "name", "email", "phone", "address", "date", "timestamp",
            "amount", "quantity", "status", "category", "description",
            "url", "ip_address", "user_agent", "session_id", "transaction_id"
        ]
        
        self.users = self._generate_users()
        self.tables = self._generate_tables()
        self.lineage = self._generate_lineage()
        self.ownership = self._generate_ownership()
        self.masked_data = self._generate_masked_data()
        
    def _generate_users(self) -> List[Dict[str, Any]]:
        """Generate synthetic users."""
        users = []
        first_names = ["Alice", "Bob", "Charlie", "Diana", "Eve", "Frank", "Grace", "Henry"]
        last_names = ["Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller"]
        
        for i in range(data_config.USERS_COUNT):
            user = {
                "user_id": f"user_{i:03d}",
                "name": f"{random.choice(first_names)} {random.choice(last_names)}",
                "email": f"user{i}@company.com",
                "department": random.choice(self.departments),
                "role": random.choice(["Data Engineer", "Data Scientist", "Analyst", "Manager"]),
                "created_date": (datetime.now() - timedelta(days=random.randint(30, 365))).isoformat(),
                "is_active": random.choice([True, False])
            }
            users.append(user)
        return users
    
    def _generate_tables(self) -> List[Dict[str, Any]]:
        """Generate synthetic table schemas."""
        tables = []
        
        for i in range(data_config.TABLES_COUNT):
            table_name = f"{random.choice(self.table_prefixes)}{random.choice(self.data_types)}_{i:03d}"
            
            # Generate columns
            columns = []
            for j in range(random.randint(5, data_config.COLUMNS_PER_TABLE)):
                col_type = random.choice(self.column_types)
                column = {
                    "column_id": f"col_{i:03d}_{j:03d}",
                    "name": f"{col_type}_{j}",
                    "data_type": self._get_data_type(col_type),
                    "is_nullable": random.choice([True, False]),
                    "is_primary_key": j == 0,  # First column is usually PK
                    "description": f"Column containing {col_type} data"
                }
                columns.append(column)
            
            table = {
                "table_id": f"table_{i:03d}",
                "name": table_name,
                "schema": "public",
                "database": "data_warehouse",
                "columns": columns,
                "row_count": random.randint(1000, 1000000),
                "size_mb": random.randint(10, 10000),
                "created_date": (datetime.now() - timedelta(days=random.randint(100, 1000))).isoformat(),
                "last_updated": (datetime.now() - timedelta(days=random.randint(1, 30))).isoformat(),
                "description": f"Table containing {random.choice(self.data_types)} data"
            }
            tables.append(table)
        
        return tables
    
    def _get_data_type(self, column_type: str) -> str:
        """Map column type to SQL data type."""
        type_mapping = {
            "id": "BIGINT",
            "name": "VARCHAR(255)",
            "email": "VARCHAR(100)",
            "phone": "VARCHAR(20)",
            "address": "TEXT",
            "date": "DATE",
            "timestamp": "TIMESTAMP",
            "amount": "DECIMAL(10,2)",
            "quantity": "INTEGER",
            "status": "VARCHAR(50)",
            "category": "VARCHAR(100)",
            "description": "TEXT",
            "url": "VARCHAR(500)",
            "ip_address": "VARCHAR(45)",
            "user_agent": "TEXT",
            "session_id": "VARCHAR(100)",
            "transaction_id": "VARCHAR(100)"
        }
        return type_mapping.get(column_type, "VARCHAR(255)")
    
    def _generate_lineage(self) -> List[Dict[str, Any]]:
        """Generate data lineage relationships."""
        lineage = []
        
        # Create parent-child relationships between tables
        for i, table in enumerate(self.tables):
            # Each table can have 0-3 parent tables
            num_parents = random.randint(0, min(3, i))
            parents = random.sample(self.tables[:i], num_parents) if i > 0 else []
            
            for parent in parents:
                lineage_item = {
                    "lineage_id": f"lineage_{len(lineage):04d}",
                    "source_table": parent["table_id"],
                    "target_table": table["table_id"],
                    "transformation_type": random.choice([
                        "aggregation", "filtering", "joining", "mapping", "cleaning"
                    ]),
                    "transformation_logic": f"SELECT * FROM {parent['name']} WHERE condition",
                    "created_date": (datetime.now() - timedelta(days=random.randint(1, 100))).isoformat(),
                    "created_by": random.choice(self.users)["user_id"]
                }
                lineage.append(lineage_item)
        
        return lineage
    
    def _generate_ownership(self) -> List[Dict[str, Any]]:
        """Generate data ownership assignments."""
        ownership = []
        
        for table in self.tables:
            # Primary owner
            primary_owner = random.choice(self.users)
            ownership_item = {
                "ownership_id": f"ownership_{len(ownership):04d}",
                "table_id": table["table_id"],
                "owner_id": primary_owner["user_id"],
                "role": "primary_owner",
                "assigned_date": (datetime.now() - timedelta(days=random.randint(1, 200))).isoformat(),
                "department": primary_owner["department"]
            }
            ownership.append(ownership_item)
            
            # Secondary owner (sometimes)
            if random.random() < 0.3:
                secondary_owner = random.choice([u for u in self.users if u["user_id"] != primary_owner["user_id"]])
                ownership_item = {
                    "ownership_id": f"ownership_{len(ownership):04d}",
                    "table_id": table["table_id"],
                    "owner_id": secondary_owner["user_id"],
                    "role": "secondary_owner",
                    "assigned_date": (datetime.now() - timedelta(days=random.randint(1, 200))).isoformat(),
                    "department": secondary_owner["department"]
                }
                ownership.append(ownership_item)
        
        return ownership
    
    def _generate_masked_data(self) -> List[Dict[str, Any]]:
        """Generate masked data examples."""
        masked_data = []
        
        for table in self.tables:
            # Generate 1-3 masked columns per table
            num_masked = random.randint(1, 3)
            masked_columns = random.sample(table["columns"], min(num_masked, len(table["columns"])))
            
            for column in masked_columns:
                if column["name"] in ["email", "phone", "address", "ip_address"]:
                    masked_item = {
                        "masked_id": f"masked_{len(masked_data):04d}",
                        "table_id": table["table_id"],
                        "column_id": column["column_id"],
                        "original_value": self._generate_original_value(column["name"]),
                        "masked_value": self._generate_masked_value(column["name"]),
                        "masking_type": random.choice(["hash", "encryption", "redaction", "substitution"]),
                        "masked_date": (datetime.now() - timedelta(days=random.randint(1, 50))).isoformat(),
                        "masked_by": random.choice(self.users)["user_id"]
                    }
                    masked_data.append(masked_item)
        
        return masked_data
    
    def _generate_original_value(self, column_name: str) -> str:
        """Generate original value for masking."""
        if "email" in column_name:
            return f"user{random.randint(1, 1000)}@example.com"
        elif "phone" in column_name:
            return f"+1-555-{random.randint(100, 999)}-{random.randint(1000, 9999)}"
        elif "address" in column_name:
            return f"{random.randint(100, 9999)} Main St, City, State {random.randint(10000, 99999)}"
        elif "ip_address" in column_name:
            return f"192.168.{random.randint(1, 255)}.{random.randint(1, 255)}"
        else:
            return f"original_value_{random.randint(1, 1000)}"
    
    def _generate_masked_value(self, column_name: str) -> str:
        """Generate masked value."""
        if "email" in column_name:
            return f"***@***.***"
        elif "phone" in column_name:
            return f"+1-***-***-****"
        elif "address" in column_name:
            return f"*** *** St, ***, ** *****"
        elif "ip_address" in column_name:
            return f"***.***.***.***"
        else:
            return f"masked_value_{random.randint(1, 1000)}"
    
    def generate_documents(self) -> List[Dict[str, Any]]:
        """Generate documents for the RAG system."""
        documents = []
        
        # Add table documents
        for table in self.tables:
            doc = {
                "id": f"table_{table['table_id']}",
                "content": f"Table {table['name']} in schema {table['schema']} contains {table['row_count']} rows. {table['description']}",
                "metadata": {
                    "type": "table",
                    "table_id": table["table_id"],
                    "schema": table["schema"],
                    "database": table["database"],
                    "row_count": table["row_count"],
                    "size_mb": table["size_mb"],
                    "created_date": table["created_date"],
                    "last_updated": table["last_updated"]
                }
            }
            documents.append(doc)
        
        # Add column documents
        for table in self.tables:
            for column in table["columns"]:
                doc = {
                    "id": f"column_{column['column_id']}",
                    "content": f"Column {column['name']} in table {table['name']} is of type {column['data_type']}. {column['description']}",
                    "metadata": {
                        "type": "column",
                        "column_id": column["column_id"],
                        "table_id": table["table_id"],
                        "table_name": table["name"],
                        "data_type": column["data_type"],
                        "is_nullable": column["is_nullable"],
                        "is_primary_key": column["is_primary_key"]
                    }
                }
                documents.append(doc)
        
        # Add ownership documents
        for ownership in self.ownership:
            user = next(u for u in self.users if u["user_id"] == ownership["owner_id"])
            table = next(t for t in self.tables if t["table_id"] == ownership["table_id"])
            doc = {
                "id": f"ownership_{ownership['ownership_id']}",
                "content": f"{user['name']} ({user['role']}) in {user['department']} department is the {ownership['role']} of table {table['name']}",
                "metadata": {
                    "type": "ownership",
                    "ownership_id": ownership["ownership_id"],
                    "table_id": ownership["table_id"],
                    "owner_id": ownership["owner_id"],
                    "owner_name": user["name"],
                    "owner_role": user["role"],
                    "owner_department": user["department"],
                    "ownership_role": ownership["role"]
                }
            }
            documents.append(doc)
        
        # Add lineage documents
        for lineage in self.lineage:
            source_table = next(t for t in self.tables if t["table_id"] == lineage["source_table"])
            target_table = next(t for t in self.tables if t["table_id"] == lineage["target_table"])
            doc = {
                "id": f"lineage_{lineage['lineage_id']}",
                "content": f"Table {source_table['name']} feeds into {target_table['name']} through {lineage['transformation_type']} transformation",
                "metadata": {
                    "type": "lineage",
                    "lineage_id": lineage["lineage_id"],
                    "source_table": lineage["source_table"],
                    "target_table": lineage["target_table"],
                    "transformation_type": lineage["transformation_type"],
                    "created_by": lineage["created_by"]
                }
            }
            documents.append(doc)
        
        return documents
    
    def save_data(self):
        """Save all generated data to files."""
        os.makedirs(data_config.DATA_DIR, exist_ok=True)
        
        # Save as JSON files
        data_files = {
            "users.json": self.users,
            "tables.json": self.tables,
            "lineage.json": self.lineage,
            "ownership.json": self.ownership,
            "masked_data.json": self.masked_data,
            "documents.json": self.generate_documents()
        }
        
        for filename, data in data_files.items():
            filepath = os.path.join(data_config.DATA_DIR, filename)
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2)
            print(f"Generated {filename} with {len(data)} records")
        
        # Save as CSV for easier analysis
        pd.DataFrame(self.users).to_csv(os.path.join(data_config.DATA_DIR, "users.csv"), index=False)
        pd.DataFrame(self.ownership).to_csv(os.path.join(data_config.DATA_DIR, "ownership.csv"), index=False)
        pd.DataFrame(self.lineage).to_csv(os.path.join(data_config.DATA_DIR, "lineage.csv"), index=False)
        pd.DataFrame(self.masked_data).to_csv(os.path.join(data_config.DATA_DIR, "masked_data.csv"), index=False)
        
        print(f"\nData generation complete! Files saved to {data_config.DATA_DIR}/")
        print(f"Generated {len(self.users)} users, {len(self.tables)} tables, {len(self.lineage)} lineage relationships")

def main():
    """Main function to generate synthetic data."""
    print("Generating synthetic data governance datasets...")
    
    generator = SyntheticDataGenerator()
    generator.save_data()
    
    print("\nData generation summary:")
    print(f"- Users: {len(generator.users)}")
    print(f"- Tables: {len(generator.tables)}")
    print(f"- Lineage relationships: {len(generator.lineage)}")
    print(f"- Ownership records: {len(generator.ownership)}")
    print(f"- Masked data records: {len(generator.masked_data)}")
    print(f"- Total documents for RAG: {len(generator.generate_documents())}")

if __name__ == "__main__":
    main() 