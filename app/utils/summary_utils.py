import json
import logging
import os


def generate_and_save_summary(processors, skip_neo4j=False):
    """
    Generate summary statistics and save them to a JSON file.

    Args:
        processors: Dictionary of processor instances
        skip_neo4j: Whether Neo4j processing was skipped
    """
    logger = logging.getLogger(__name__)
    summary = {}

    # Make sure the logs directory exists
    os.makedirs("logs", exist_ok=True)

    # Extract processors
    reg_processor = processors.get("reg_processor")
    scan_processor = processors.get("scan_processor")
    session_processor = processors.get("session_processor")

    # Create summary statistics for registration data
    if reg_processor and hasattr(reg_processor, "df_reg_demo_this"):
        reg_summary = {
            "total_records": {
                "this_year": len(reg_processor.df_reg_demo_this),
                "last_year_bva": len(reg_processor.df_reg_demo_last_bva),
                "last_year_lva": len(reg_processor.df_reg_demo_last_lva),
            }
        }
        
        # Only add job_roles for veterinary events (vet-specific)
        if "job_role" in reg_processor.df_reg_demo_this.columns:
            reg_summary["job_roles"] = reg_processor.df_reg_demo_this["job_role"].value_counts().to_dict()
            logger.info("Added job role summary (veterinary event)")
        else:
            logger.info("No job_role column found - skipping job role summary (generic event)")
        
        # Try to find specialization columns (different for vet vs generic events)
        specialization_columns = [
            "what_type_does_your_practice_specialise_in",  # BVA/vet-specific
            "what_best_describes_what_you_do",  # ECOMM-specific
            "specialization_current",  # Generic fallback
            "main_specialization"  # Generic fallback
        ]
        
        specialization_found = False
        for col in specialization_columns:
            if col in reg_processor.df_reg_demo_this.columns:
                reg_summary["specializations"] = reg_processor.df_reg_demo_this[col].value_counts().to_dict()
                logger.info(f"Using specialization column: {col}")
                specialization_found = True
                break
        
        if not specialization_found:
            logger.info("No specialization column found - skipping specialization summary")
            reg_summary["specializations"] = {}
        
        summary["registration"] = reg_summary

    # Create summary statistics for scan data if available
    if (
        scan_processor
        and hasattr(scan_processor, "enhanced_seminars_df_bva")
        and hasattr(scan_processor, "enhanced_seminars_df_lva")
    ):
        scan_summary = {
            "total_scans": {
                "last_year_bva": len(scan_processor.enhanced_seminars_df_bva),
                "last_year_lva": len(scan_processor.enhanced_seminars_df_lva),
            },
            "unique_seminars": {
                "last_year_bva": len(
                    scan_processor.seminars_scans_past_enhanced_bva[
                        "Seminar Name"
                    ].unique()
                ),
                "last_year_lva": len(
                    scan_processor.seminars_scans_past_enhanced_lva[
                        "Seminar Name"
                    ].unique()
                ),
            },
            "unique_attendees": {
                "last_year_bva": len(
                    scan_processor.enhanced_seminars_df_bva["Badge Id"].unique()
                ),
                "last_year_lva": len(
                    scan_processor.enhanced_seminars_df_lva["Badge Id"].unique()
                ),
            },
        }
        summary["scan"] = scan_summary

    # Create summary statistics for session data if available
    if (
        session_processor
        and hasattr(session_processor, "session_this_filtered_valid_cols")
        and hasattr(session_processor, "streams_catalog")
    ):
        session_summary = {
            "total_sessions": {
                "this_year": len(session_processor.session_this_filtered_valid_cols),
                "last_year_bva": len(
                    session_processor.session_last_filtered_valid_cols_bva
                ),
                "last_year_lva": len(
                    session_processor.session_last_filtered_valid_cols_lva
                ),
            },
            "unique_streams": len(session_processor.streams),
            "stream_categories": list(session_processor.streams_catalog.keys()),
        }
        summary["session"] = session_summary

    # Add Neo4j statistics if available and Neo4j processing wasn't skipped
    if not skip_neo4j:
        add_neo4j_statistics(summary, processors)

    # Add session recommendation statistics if available
    session_recommendation_processor = processors.get(
        "session_recommendation_processor"
    )
    if session_recommendation_processor and hasattr(
        session_recommendation_processor, "statistics"
    ):
        session_recommendation_summary = session_recommendation_processor.statistics
        summary["session_recommendation"] = session_recommendation_summary

    # Save summary as JSON
    with open("logs/processing_summary.json", "w") as f:
        json.dump(summary, f, indent=4, default=str)

    # Log and print summary information
    print_summary_statistics(summary, skip_neo4j, reg_processor)

    return summary


def add_neo4j_statistics(summary, processors):
    """
    Add Neo4j statistics to the summary dictionary.

    Args:
        summary: Summary dictionary to update
        processors: Dictionary of processor instances
    """
    # Add Neo4j visitor statistics if available
    neo4j_visitor_processor = processors.get("neo4j_visitor_processor")
    if neo4j_visitor_processor and hasattr(neo4j_visitor_processor, "statistics"):
        neo4j_visitor_summary = {
            "nodes_created": neo4j_visitor_processor.statistics["nodes_created"],
            "nodes_skipped": neo4j_visitor_processor.statistics["nodes_skipped"],
            "total_nodes_created": sum(
                neo4j_visitor_processor.statistics["nodes_created"].values()
            ),
            "total_nodes_skipped": sum(
                neo4j_visitor_processor.statistics["nodes_skipped"].values()
            ),
        }
        summary["neo4j_visitor"] = neo4j_visitor_summary

    # Add Neo4j session statistics if available
    neo4j_session_processor = processors.get("neo4j_session_processor")
    if neo4j_session_processor and hasattr(neo4j_session_processor, "statistics"):
        neo4j_session_summary = {
            "nodes_created": neo4j_session_processor.statistics["nodes_created"],
            "nodes_skipped": neo4j_session_processor.statistics["nodes_skipped"],
            "relationships_created": neo4j_session_processor.statistics[
                "relationships_created"
            ],
            "relationships_skipped": neo4j_session_processor.statistics[
                "relationships_skipped"
            ],
            "total_nodes_created": sum(
                neo4j_session_processor.statistics["nodes_created"].values()
            ),
            "total_nodes_skipped": sum(
                neo4j_session_processor.statistics["nodes_skipped"].values()
            ),
            "total_relationships_created": sum(
                neo4j_session_processor.statistics["relationships_created"].values()
            ),
            "total_relationships_skipped": sum(
                neo4j_session_processor.statistics["relationships_skipped"].values()
            ),
        }
        summary["neo4j_session"] = neo4j_session_summary

    # Add Neo4j job stream statistics if available
    neo4j_job_stream_processor = processors.get("neo4j_job_stream_processor")
    if neo4j_job_stream_processor and hasattr(neo4j_job_stream_processor, "statistics"):
        neo4j_job_stream_summary = neo4j_job_stream_processor.statistics
        summary["neo4j_job_stream"] = neo4j_job_stream_summary

    # Add Neo4j specialization stream statistics if available
    neo4j_specialization_stream_processor = processors.get(
        "neo4j_specialization_stream_processor"
    )
    if neo4j_specialization_stream_processor and hasattr(
        neo4j_specialization_stream_processor, "statistics"
    ):
        neo4j_specialization_stream_summary = (
            neo4j_specialization_stream_processor.statistics
        )
        # Calculate totals
        neo4j_specialization_stream_summary["total_relationships_created"] = sum(
            neo4j_specialization_stream_summary["relationships_created"].values()
        )
        neo4j_specialization_stream_summary["total_relationships_skipped"] = sum(
            neo4j_specialization_stream_summary["relationships_skipped"].values()
        )
        neo4j_specialization_stream_summary["total_visitor_nodes_processed"] = sum(
            neo4j_specialization_stream_summary["visitor_nodes_processed"].values()
        )
        summary["neo4j_specialization_stream"] = neo4j_specialization_stream_summary

    # Add Neo4j visitor relationship statistics if available
    neo4j_visitor_relationship_processor = processors.get(
        "neo4j_visitor_relationship_processor"
    )
    if neo4j_visitor_relationship_processor and hasattr(
        neo4j_visitor_relationship_processor, "statistics"
    ):
        neo4j_visitor_relationship_summary = (
            neo4j_visitor_relationship_processor.statistics
        )
        # Calculate totals
        neo4j_visitor_relationship_summary["total_relationships_created"] = sum(
            neo4j_visitor_relationship_summary["relationships_created"].values()
        )
        neo4j_visitor_relationship_summary["total_relationships_skipped"] = sum(
            neo4j_visitor_relationship_summary["relationships_skipped"].values()
        )
        summary["neo4j_visitor_relationship"] = neo4j_visitor_relationship_summary

    # Add Session Embedding statistics if available
    session_embedding_processor = processors.get("session_embedding_processor")
    if session_embedding_processor and hasattr(
        session_embedding_processor, "statistics"
    ):
        session_embedding_summary = session_embedding_processor.statistics
        summary["session_embedding"] = session_embedding_summary


def print_summary_statistics(summary, skip_neo4j, reg_processor=None):
    """
    Print summary statistics to console and log key information.

    Args:
        summary: Dictionary containing summary statistics
        skip_neo4j: Whether Neo4j processing was skipped
        reg_processor: Registration processor instance (for data distributions)
    """
    logger = logging.getLogger(__name__)

    # Get event names from processor config if available
    main_event_name = "BVA"  # Default fallback
    secondary_event_name = "LVA"  # Default fallback
    
    if reg_processor and hasattr(reg_processor, 'config'):
        event_config = reg_processor.config.get("event", {})
        main_event_name = event_config.get("main_event_name", "BVA")
        secondary_event_name = event_config.get("secondary_event_name", "LVA")

    # Print registration summary
    if "registration" in summary:
        reg_summary = summary["registration"]
        logger.info(
            f"Total registration records this year: {reg_summary['total_records']['this_year']}"
        )
        logger.info(
            f"Total registration records last year {main_event_name}: {reg_summary['total_records']['last_year_bva']}"
        )
        logger.info(
            f"Total registration records last year {secondary_event_name}: {reg_summary['total_records']['last_year_lva']}"
        )

        print("\nSummary Statistics:")
        print(
            f"Total registration records this year: {reg_summary['total_records']['this_year']}"
        )
        print(
            f"Total registration records last year {main_event_name}: {reg_summary['total_records']['last_year_bva']}"
        )
        print(
            f"Total registration records last year {secondary_event_name}: {reg_summary['total_records']['last_year_lva']}"
        )

    # Print scan summary
    if "scan" in summary:
        scan_summary = summary["scan"]
        logger.info(
            f"Total seminar scans last year {main_event_name}: {scan_summary['total_scans']['last_year_bva']}"
        )
        logger.info(
            f"Total seminar scans last year {secondary_event_name}: {scan_summary['total_scans']['last_year_lva']}"
        )
        logger.info(
            f"Unique seminar attendees last year {main_event_name}: {scan_summary['unique_attendees']['last_year_bva']}"
        )
        logger.info(
            f"Unique seminar attendees last year {secondary_event_name}: {scan_summary['unique_attendees']['last_year_lva']}"
        )

        print("\nScan Summary:")
        print(
            f"Total seminar scans last year {main_event_name}: {scan_summary['total_scans']['last_year_bva']}"
        )
        print(
            f"Total seminar scans last year {secondary_event_name}: {scan_summary['total_scans']['last_year_lva']}"
        )
        print(
            f"Unique seminar attendees last year {main_event_name}: {scan_summary['unique_attendees']['last_year_bva']}"
        )
        print(
            f"Unique seminar attendees last year {secondary_event_name}: {scan_summary['unique_attendees']['last_year_lva']}"
        )

    # Print session summary
    if "session" in summary:
        session_summary = summary["session"]
        logger.info(
            f"Total sessions this year: {session_summary['total_sessions']['this_year']}"
        )
        logger.info(
            f"Total sessions last year {main_event_name}: {session_summary['total_sessions']['last_year_bva']}"
        )
        logger.info(
            f"Total sessions last year {secondary_event_name}: {session_summary['total_sessions']['last_year_lva']}"
        )
        logger.info(
            f"Total unique stream categories: {session_summary['unique_streams']}"
        )

        print("\nSession Summary:")
        print(
            f"Total sessions this year: {session_summary['total_sessions']['this_year']}"
        )
        print(
            f"Total sessions last year {main_event_name}: {session_summary['total_sessions']['last_year_bva']}"
        )
        print(
            f"Total sessions last year {secondary_event_name}: {session_summary['total_sessions']['last_year_lva']}"
        )
        print(f"Total unique stream categories: {session_summary['unique_streams']}")
        print("\nTop 5 Stream Categories:")
        for i, stream in enumerate(session_summary["stream_categories"][:5]):
            print(f"  {i+1}. {stream}")

    # Print session recommendation summary
    if "session_recommendation" in summary:
        rec_summary = summary["session_recommendation"]
        logger.info(
            f"Total visitors processed for recommendations: {rec_summary['total_visitors_processed']}"
        )
        logger.info(
            f"Visitors with successful recommendations: {rec_summary['visitors_with_recommendations']}"
        )
        logger.info(
            f"Total recommendations generated: {rec_summary['total_recommendations_generated']}"
        )
        logger.info(
            f"Unique sessions recommended: {rec_summary['unique_recommended_sessions']}"
        )
        if rec_summary.get("errors", 0) > 0:
            logger.info(f"Errors encountered: {rec_summary['errors']}")

        print("\nSession Recommendation Summary:")
        print(f"Total visitors processed: {rec_summary['total_visitors_processed']}")
        print(
            f"Visitors with successful recommendations: {rec_summary['visitors_with_recommendations']}"
        )
        print(
            f"Total recommendations generated: {rec_summary['total_recommendations_generated']}"
        )
        print(
            f"Unique sessions recommended: {rec_summary['unique_recommended_sessions']}"
        )
        print(f"Processing time: {rec_summary['processing_time']:.2f} seconds")
        if rec_summary.get("errors", 0) > 0:
            print(f"Errors encountered: {rec_summary['errors']}")

    # Print Neo4j statistics
    if not skip_neo4j:
        print_neo4j_statistics(summary, main_event_name, secondary_event_name)

    # Print data distributions - FIXED: Check if columns exist before accessing
    if reg_processor and hasattr(reg_processor, "df_reg_demo_this"):
        # Only print job role distribution if the column exists (veterinary events)
        if "job_role" in reg_processor.df_reg_demo_this.columns:
            print("\nJob Role Distribution This Year:")
            print(reg_processor.df_reg_demo_this["job_role"].value_counts())
        else:
            print("\nJob Role Distribution This Year:")
            print("Not available (generic event - no job role processing)")

        # Print specialization distribution - try multiple possible column names
        specialization_columns = [
            "what_type_does_your_practice_specialise_in",  # BVA/vet-specific
            "what_best_describes_what_you_do",  # ECOMM-specific
            "specialization_current",  # Generic fallback
            "main_specialization"  # Generic fallback
        ]
        
        specialization_printed = False
        for col in specialization_columns:
            if col in reg_processor.df_reg_demo_this.columns:
                print(f"\nPractice Specialization Distribution This Year:")
                print(reg_processor.df_reg_demo_this[col].value_counts())
                specialization_printed = True
                break
        
        if not specialization_printed:
            print(f"\nPractice Specialization Distribution This Year:")
            print("Not available (no specialization column found)")

    # Print Neo4j usage information if Neo4j processing wasn't skipped
    if not skip_neo4j:
        print_neo4j_usage_info()


def print_neo4j_statistics(summary, main_event_name="BVA", secondary_event_name="LVA"):
    """
    Print Neo4j statistics to console.

    Args:
        summary: Dictionary containing summary statistics
        main_event_name: Name of the main event (default: BVA)
        secondary_event_name: Name of the secondary event (default: LVA)
    """
    logger = logging.getLogger(__name__)

    # Print Neo4j visitor statistics
    if "neo4j_visitor" in summary:
        neo4j_visitor_summary = summary["neo4j_visitor"]
        logger.info(
            f"Total Neo4j visitor nodes created: {neo4j_visitor_summary['total_nodes_created']}"
        )
        logger.info(
            f"Total Neo4j visitor nodes skipped: {neo4j_visitor_summary['total_nodes_skipped']}"
        )

        print("\nNeo4j Visitor Upload Summary:")
        print(
            f"Total visitor nodes created: {neo4j_visitor_summary['total_nodes_created']}"
        )
        print(
            f"Total visitor nodes skipped: {neo4j_visitor_summary['total_nodes_skipped']}"
        )
        print("\nDetailed Neo4j Visitor Statistics:")
        print(
            f"  Visitors this year: {neo4j_visitor_summary['nodes_created']['visitor_this_year']} created, {neo4j_visitor_summary['nodes_skipped']['visitor_this_year']} skipped"
        )
        print(
            f"  Visitors last year {main_event_name}: {neo4j_visitor_summary['nodes_created']['visitor_last_year_bva']} created, {neo4j_visitor_summary['nodes_skipped']['visitor_last_year_bva']} skipped"
        )
        print(
            f"  Visitors last year {secondary_event_name}: {neo4j_visitor_summary['nodes_created']['visitor_last_year_lva']} created, {neo4j_visitor_summary['nodes_skipped']['visitor_last_year_lva']} skipped"
        )

    # Print Neo4j session statistics
    if "neo4j_session" in summary:
        neo4j_session_summary = summary["neo4j_session"]
        logger.info(
            f"Total Neo4j session nodes created: {neo4j_session_summary['total_nodes_created']}"
        )
        logger.info(
            f"Total Neo4j session nodes skipped: {neo4j_session_summary['total_nodes_skipped']}"
        )
        logger.info(
            f"Total Neo4j relationships created: {neo4j_session_summary['total_relationships_created']}"
        )
        logger.info(
            f"Total Neo4j relationships skipped: {neo4j_session_summary['total_relationships_skipped']}"
        )

        print("\nNeo4j Session Upload Summary:")
        print(
            f"Total session nodes created: {neo4j_session_summary['total_nodes_created']}"
        )
        print(
            f"Total session nodes skipped: {neo4j_session_summary['total_nodes_skipped']}"
        )
        print(
            f"Total relationships created: {neo4j_session_summary['total_relationships_created']}"
        )
        print(
            f"Total relationships skipped: {neo4j_session_summary['total_relationships_skipped']}"
        )
        print("\nDetailed Neo4j Session Statistics:")
        print(
            f"  Sessions this year: {neo4j_session_summary['nodes_created']['sessions_this_year']} created, {neo4j_session_summary['nodes_skipped']['sessions_this_year']} skipped"
        )
        print(
            f"  Sessions past year {main_event_name}: {neo4j_session_summary['nodes_created']['sessions_past_year_bva']} created, {neo4j_session_summary['nodes_skipped']['sessions_past_year_bva']} skipped"
        )
        print(
            f"  Sessions past year {secondary_event_name}: {neo4j_session_summary['nodes_created']['sessions_past_year_lva']} created, {neo4j_session_summary['nodes_skipped']['sessions_past_year_lva']} skipped"
        )
        print(
            f"  Stream nodes: {neo4j_session_summary['nodes_created']['streams']} created, {neo4j_session_summary['nodes_skipped']['streams']} skipped"
        )
        print(
            f"  Session-Stream relationships this year: {neo4j_session_summary['relationships_created']['sessions_this_year_has_stream']} created, {neo4j_session_summary['relationships_skipped']['sessions_this_year_has_stream']} skipped"
        )
        print(
            f"  Session-Stream relationships past year: {neo4j_session_summary['relationships_created']['sessions_past_year_has_stream']} created, {neo4j_session_summary['relationships_skipped']['sessions_past_year_has_stream']} skipped"
        )

    # Print Neo4j job stream statistics
    if "neo4j_job_stream" in summary:
        neo4j_job_stream_summary = summary["neo4j_job_stream"]
        print("\nNeo4j Job-Stream Relationship Summary:")
        print(
            f"Total job-stream relationships created: {neo4j_job_stream_summary['relationships_created']}"
        )
        print(
            f"Total job-stream relationships skipped: {neo4j_job_stream_summary['relationships_skipped']}"
        )
        print("\nDetailed Job-Stream Statistics:")
        print(
            f"  Visitor nodes processed: {neo4j_job_stream_summary['visitor_nodes_processed']}"
        )
        print(f"  Job roles matched: {neo4j_job_stream_summary['job_roles_processed']}")
        print(
            f"  Stream matches found: {neo4j_job_stream_summary['stream_matches_found']}"
        )

    # Print Neo4j specialization stream statistics
    if "neo4j_specialization_stream" in summary:
        neo4j_specialization_stream_summary = summary["neo4j_specialization_stream"]
        print("\nNeo4j Specialization-Stream Relationship Summary:")
        print(
            f"Total specialization-stream relationships created: {neo4j_specialization_stream_summary['total_relationships_created']}"
        )
        print(
            f"Total specialization-stream relationships skipped: {neo4j_specialization_stream_summary['total_relationships_skipped']}"
        )
        print("\nDetailed Specialization-Stream Statistics:")
        print(f"  Visitor nodes processed:")
        print(
            f"    This year: {neo4j_specialization_stream_summary['visitor_nodes_processed']['visitor_this_year']}"
        )
        print(
            f"    Last year {main_event_name}: {neo4j_specialization_stream_summary['visitor_nodes_processed']['visitor_last_year_bva']}"
        )
        print(
            f"    Last year {secondary_event_name}: {neo4j_specialization_stream_summary['visitor_nodes_processed']['visitor_last_year_lva']}"
        )
        print(
            f"  Specializations processed: {neo4j_specialization_stream_summary['specializations_processed']}"
        )
        print(
            f"  Stream matches found: {neo4j_specialization_stream_summary['stream_matches_found']}"
        )

    # Print Neo4j visitor relationship statistics
    if "neo4j_visitor_relationship" in summary:
        neo4j_visitor_relationship_summary = summary["neo4j_visitor_relationship"]
        print("\nNeo4j Visitor Relationship Summary:")
        print(
            f"Total visitor relationships created: {neo4j_visitor_relationship_summary['total_relationships_created']}"
        )
        print(
            f"Total visitor relationships skipped: {neo4j_visitor_relationship_summary['total_relationships_skipped']}"
        )
        print("\nDetailed Visitor Relationship Statistics:")
        print(
            f"  Same_Visitor relationships for {main_event_name}: {neo4j_visitor_relationship_summary['relationships_created']['same_visitor_bva']} created, {neo4j_visitor_relationship_summary['relationships_skipped']['same_visitor_bva']} skipped"
        )
        print(
            f"  Same_Visitor relationships for {secondary_event_name}: {neo4j_visitor_relationship_summary['relationships_created']['same_visitor_lva']} created, {neo4j_visitor_relationship_summary['relationships_skipped']['same_visitor_lva']} skipped"
        )
        print(
            f"  attended_session relationships for {main_event_name}: {neo4j_visitor_relationship_summary['relationships_created']['attended_session_bva']} created, {neo4j_visitor_relationship_summary['relationships_skipped']['attended_session_bva']} skipped"
        )
        print(
            f"  attended_session relationships for {secondary_event_name}: {neo4j_visitor_relationship_summary['relationships_created']['attended_session_lva']} created, {neo4j_visitor_relationship_summary['relationships_skipped']['attended_session_lva']} skipped"
        )

    # Print session embedding statistics
    if "session_embedding" in summary:
        session_embedding_summary = summary["session_embedding"]
        logger.info(
            f"Total sessions processed for embeddings: {session_embedding_summary['total_sessions_processed']}"
        )
        logger.info(
            f"Sessions with embeddings created: {session_embedding_summary['sessions_with_embeddings']}"
        )

        print("\nSession Embedding Summary:")
        print(
            f"Total sessions processed: {session_embedding_summary['total_sessions_processed']}"
        )
        print(
            f"Sessions with embeddings created: {session_embedding_summary['sessions_with_embeddings']}"
        )
        print(
            f"Sessions with stream descriptions included: {session_embedding_summary['sessions_with_stream_descriptions']}"
        )
        print("\nDetailed Session Embedding Statistics:")
        print(
            f"  Sessions this year: {session_embedding_summary['sessions_by_type']['sessions_this_year']}"
        )
        print(
            f"  Sessions past year: {session_embedding_summary['sessions_by_type']['sessions_past_year']}"
        )
        if session_embedding_summary.get("errors", 0) > 0:
            print(f"  Errors encountered: {session_embedding_summary['errors']}")


def print_neo4j_usage_info():
    """
    Print Neo4j usage information and example queries.
    """
    print("\nNeo4j Processing Overview:")
    print(
        "- Uploaded visitor data to Neo4j as separate node types for this year and past years"
    )
    print(
        "- Uploaded session data to Neo4j as separate node types for this year and past years"
    )
    print("- Created stream nodes with descriptions")
    print("- Established HAS_STREAM relationships between sessions and streams")
    print(
        "- Created JOB_TO_STREAM relationships between visitors and recommended streams based on job roles"
    )
    print(
        "- Created SPECIALIZATION_TO_STREAM relationships between visitors and relevant streams based on practice specializations"
    )
    print(
        "- Created Same_Visitor relationships between visitors this year and visitors from past events"
    )
    print(
        "- Created attended_session relationships between past visitors and the sessions they attended"
    )
    print("- Generated text embeddings for all session nodes to enable semantic search")
    print("- Generated personalized session recommendations for all visitors")
    print("- Use the Neo4j Browser to explore and query the data")
    print("- Example queries:")
    print(
        "  1. MATCH (s:Sessions_this_year)-[:HAS_STREAM]->(st:Stream) RETURN s, st LIMIT 25"
    )
    print(
        "  2. MATCH (v:Visitor_this_year)-[:JOB_TO_STREAM]->(st:Stream) RETURN v.job_role, collect(st.stream) as recommended_streams LIMIT 10"
    )
    print(
        "  3. MATCH (v:Visitor_this_year)-[:SPECIALIZATION_TO_STREAM]->(st:Stream) RETURN v.what_type_does_your_practice_specialise_in, collect(st.stream) as relevant_streams LIMIT 10"
    )
    print(
        "  4. MATCH (v:Visitor_this_year) WHERE v.job_role='Veterinary Surgeon' RETURN v.job_role, count(v) as vet_count"
    )
    print(
        "  5. MATCH (v:Visitor_this_year)-[:Same_Visitor]->(v_past) RETURN v.BadgeId, v_past.BadgeId, v_past.job_role LIMIT 10"
    )
    print(
        "  6. MATCH (v:Visitor_last_year_bva)-[:attended_session]->(s:Sessions_past_year) RETURN v.BadgeId, count(s) as sessions_attended ORDER BY sessions_attended DESC LIMIT 10"
    )
    print(
        "  7. MATCH (s:Sessions_this_year) WHERE s.embedding IS NOT NULL RETURN count(s) as sessions_with_embeddings"
    )