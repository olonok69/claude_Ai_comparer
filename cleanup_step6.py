#!/usr/bin/env python3
"""
Cleanup Script for Step 6 - Remove Job Stream Relationships

This script removes all job_to_stream relationships created by Step 6.
"""

from neo4j import GraphDatabase
from dotenv import load_dotenv
import os


def cleanup_step6_relationships():
    """Remove all job_to_stream relationships"""
    print("🧹 STEP 6 CLEANUP")
    print("=" * 20)
    
    # Load Neo4j connection
    load_dotenv("keys/.env")
    uri = os.getenv("NEO4J_URI")
    username = os.getenv("NEO4J_USERNAME")
    password = os.getenv("NEO4J_PASSWORD")
    
    if not all([uri, username, password]):
        print("❌ Missing Neo4j credentials")
        return False
    
    driver = GraphDatabase.driver(uri, auth=(username, password))
    
    try:
        with driver.session() as session:
            # Count existing relationships
            count_result = session.run("""
                MATCH (v:Visitor_this_year)-[r:job_to_stream]->(s:Stream)
                RETURN count(r) as total_relationships
            """)
            initial_count = count_result.single()["total_relationships"]
            
            if initial_count == 0:
                print("✅ No job_to_stream relationships found - database is clean")
                return True
            
            print(f"🔍 Found {initial_count} job_to_stream relationships")
            
            # Show breakdown by show
            breakdown_result = session.run("""
                MATCH (v:Visitor_this_year)-[r:job_to_stream]->(s:Stream)
                RETURN v.show as visitor_show, s.show as stream_show, count(r) as count
                ORDER BY v.show, s.show
            """)
            
            print("📊 Breakdown by show:")
            for record in breakdown_result:
                visitor_show = record["visitor_show"]
                stream_show = record["stream_show"]
                count = record["count"]
                print(f"  - {visitor_show} visitors → {stream_show} streams: {count} relationships")
            
            # Ask for confirmation
            response = input(f"\n❓ Delete all {initial_count} job_to_stream relationships? (y/N): ")
            
            if response.lower() != 'y':
                print("❌ Cleanup cancelled")
                return False
            
            # Delete all job_to_stream relationships
            delete_result = session.run("""
                MATCH (v:Visitor_this_year)-[r:job_to_stream]->(s:Stream)
                DELETE r
                RETURN count(r) as deleted_count
            """)
            
            deleted_count = delete_result.single()["deleted_count"]
            
            # Verify cleanup
            verify_result = session.run("""
                MATCH (v:Visitor_this_year)-[r:job_to_stream]->(s:Stream)
                RETURN count(r) as remaining_count
            """)
            remaining_count = verify_result.single()["remaining_count"]
            
            print(f"✅ Deleted {deleted_count} job_to_stream relationships")
            print(f"✅ Remaining relationships: {remaining_count}")
            
            if remaining_count == 0:
                print("🎉 Database is now clean - ready for Step 6 testing!")
                return True
            else:
                print("⚠️  Some relationships may still exist")
                return False
                
    except Exception as e:
        print(f"❌ Error during cleanup: {e}")
        return False
    finally:
        driver.close()


if __name__ == "__main__":
    success = cleanup_step6_relationships()
    print(f"\n{'✅ SUCCESS' if success else '❌ FAILED'}: Step 6 cleanup {'completed' if success else 'failed'}")