#!/usr/bin/env python3
"""
Gene Database Search Tool for RedDust Reclaimer Project
====================================================

This module searches NCBI and UniProt databases for optimal perchlorate reductase (pcrA/pcrB)
and chlorite dismutase (cld) genes from psychrotolerant and halotolerant organisms suitable
for Mars environment adaptation.

Author: RedDust Reclaimer AI
Date: 2024
License: MIT
"""

import requests
import time
import json
import xml.etree.ElementTree as ET
from Bio import Entrez, SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from pathlib import Path
import pandas as pd
import logging
from typing import List, Dict, Tuple, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class GeneSearchEngine:
    """
    Comprehensive gene search engine for identifying optimal perchlorate detoxification genes
    from organisms adapted to extreme environments similar to Mars.
    """

    def __init__(self, email: str = "researcher@example.com"):
        """
        Initialize the gene search engine.

        Args:
            email: Email address for NCBI API access (required by NCBI)
        """
        self.email = email
        Entrez.email = email
        self.target_genes = {
            "pcrA": "perchlorate reductase subunit A",
            "pcrB": "perchlorate reductase subunit B",
            "cld": "chlorite dismutase",
        }

        # Target organisms - psychrotolerant/halotolerant bacteria
        self.target_organisms = [
            "Dechloromonas aromatica",
            "Azospira suillum",
            "Dechloromonas agitata",
            "Ideonella dechloratans",
            "Pseudomonas chloritidismutans",
            "Psychrobacter",
            "Colwellia",
            "Shewanella",
            "Planococcus",
            "Sporosarcina",
            "Bacillus psychrodurans",
            "Arthrobacter",
        ]

        self.results_dir = Path("../../data/genes")
        self.results_dir.mkdir(parents=True, exist_ok=True)

    def search_ncbi_nucleotide(
        self, gene_name: str, organism: str = None, max_results: int = 50
    ) -> List[Dict]:
        """
        Search NCBI nucleotide database for gene sequences.

        Args:
            gene_name: Name/description of the gene to search for
            organism: Optional organism name to restrict search
            max_results: Maximum number of results to return

        Returns:
            List of dictionaries containing gene information
        """
        logger.info(f"Searching NCBI nucleotide database for {gene_name}")

        # Construct search query
        query = f'"{gene_name}"[Gene Name] OR "{gene_name}"[Title]'
        if organism:
            query += f' AND "{organism}"[Organism]'

        query += " AND biomol_genomic[PROP] AND refseq[FILTER]"

        try:
            # Search for IDs
            search_handle = Entrez.esearch(
                db="nucleotide", term=query, retmax=max_results, sort="relevance"
            )
            search_results = Entrez.read(search_handle)
            search_handle.close()

            if not search_results["IdList"]:
                logger.warning(f"No results found for {gene_name} in {organism}")
                return []

            # Fetch detailed information
            ids = search_results["IdList"]
            fetch_handle = Entrez.efetch(
                db="nucleotide", id=",".join(ids), rettype="gb", retmode="xml"
            )

            records = []
            for record in Entrez.parse(fetch_handle):
                try:
                    record_info = {
                        "id": record.get("GBSeq_primary-accession", ""),
                        "definition": record.get("GBSeq_definition", ""),
                        "organism": record.get("GBSeq_organism", ""),
                        "length": int(record.get("GBSeq_length", 0)),
                        "sequence": record.get("GBSeq_sequence", ""),
                        "create_date": record.get("GBSeq_create-date", ""),
                        "gene_name": gene_name,
                        "source": "NCBI_nucleotide",
                    }

                    # Extract additional features
                    features = record.get("GBSeq_feature-table", [])
                    for feature in features:
                        if feature.get("GBFeature_key") == "CDS":
                            qualifiers = feature.get("GBFeature_quals", [])
                            for qual in qualifiers:
                                if qual.get("GBQualifier_name") == "product":
                                    record_info["product"] = qual.get(
                                        "GBQualifier_value", ""
                                    )
                                elif qual.get("GBQualifier_name") == "gene":
                                    record_info["gene_symbol"] = qual.get(
                                        "GBQualifier_value", ""
                                    )

                    records.append(record_info)

                except Exception as e:
                    logger.warning(f"Error parsing record: {e}")
                    continue

            fetch_handle.close()
            return records

        except Exception as e:
            logger.error(f"Error searching NCBI: {e}")
            return []

    def search_uniprot(
        self, gene_name: str, organism: str = None, max_results: int = 50
    ) -> List[Dict]:
        """
        Search UniProt database for protein sequences.

        Args:
            gene_name: Name/description of the gene/protein to search for
            organism: Optional organism name to restrict search
            max_results: Maximum number of results to return

        Returns:
            List of dictionaries containing protein information
        """
        logger.info(f"Searching UniProt database for {gene_name}")

        # Construct search query
        query = f'"{gene_name}" OR gene:"{gene_name}" OR protein_name:"{gene_name}"'
        if organism:
            query += f' AND organism:"{organism}"'

        # Add filters for reviewed entries and complete proteomes
        query += " AND reviewed:true"

        try:
            url = "https://rest.uniprot.org/uniprotkb/search"
            params = {
                "query": query,
                "format": "json",
                "size": max_results,
                "fields": "accession,id,gene_names,protein_name,organism_name,length,sequence,cc_function,cc_subcellular_location,cc_temperature_dependence",
            }

            response = requests.get(url, params=params)
            response.raise_for_status()

            data = response.json()
            records = []

            for entry in data.get("results", []):
                try:
                    record_info = {
                        "id": entry.get("primaryAccession", ""),
                        "entry_id": entry.get("uniProtKBId", ""),
                        "gene_names": (
                            entry.get("genes", [{}])[0]
                            .get("geneName", {})
                            .get("value", "")
                            if entry.get("genes")
                            else ""
                        ),
                        "protein_name": entry.get("proteinDescription", {})
                        .get("recommendedName", {})
                        .get("fullName", {})
                        .get("value", ""),
                        "organism": entry.get("organism", {}).get("scientificName", ""),
                        "length": entry.get("sequence", {}).get("length", 0),
                        "sequence": entry.get("sequence", {}).get("value", ""),
                        "gene_name": gene_name,
                        "source": "UniProt",
                    }

                    # Extract functional annotations
                    comments = entry.get("comments", [])
                    for comment in comments:
                        if comment.get("commentType") == "FUNCTION":
                            record_info["function"] = comment.get("texts", [{}])[0].get(
                                "value", ""
                            )
                        elif comment.get("commentType") == "SUBCELLULAR LOCATION":
                            record_info["subcellular_location"] = comment.get(
                                "texts", [{}]
                            )[0].get("value", "")
                        elif comment.get("commentType") == "TEMPERATURE DEPENDENCE":
                            record_info["temperature_dependence"] = comment.get(
                                "texts", [{}]
                            )[0].get("value", "")

                    records.append(record_info)

                except Exception as e:
                    logger.warning(f"Error parsing UniProt entry: {e}")
                    continue

            return records

        except Exception as e:
            logger.error(f"Error searching UniProt: {e}")
            return []

    def score_gene_suitability(self, record: Dict) -> float:
        """
        Score gene/protein suitability for Mars environment based on various criteria.

        Args:
            record: Gene/protein record dictionary

        Returns:
            Suitability score (0-100)
        """
        score = 0.0

        # Organism-based scoring (psychrotolerant/halotolerant organisms get higher scores)
        organism = record.get("organism", "").lower()
        for target_org in self.target_organisms:
            if target_org.lower() in organism:
                score += 30
                break

        # Cold/stress keywords in descriptions
        cold_keywords = [
            "psychro",
            "cold",
            "antarctic",
            "arctic",
            "low temperature",
            "freeze",
            "frost",
        ]
        salt_keywords = ["halo", "salt", "saline", "osmotic", "high salt"]
        stress_keywords = ["stress", "extreme", "harsh", "resistant", "tolerance"]

        description_text = (
            record.get("definition", "")
            + " "
            + record.get("function", "")
            + " "
            + record.get("temperature_dependence", "")
        ).lower()

        for keyword in cold_keywords:
            if keyword in description_text:
                score += 15

        for keyword in salt_keywords:
            if keyword in description_text:
                score += 10

        for keyword in stress_keywords:
            if keyword in description_text:
                score += 10

        # Sequence length (prefer complete, reasonable-sized proteins)
        length = record.get("length", 0)
        if 200 <= length <= 1500:  # Reasonable protein size
            score += 15
        elif 100 <= length < 200 or 1500 < length <= 2000:
            score += 10
        elif length > 0:
            score += 5

        # RefSeq entries get bonus points (higher quality)
        if "refseq" in record.get("id", "").lower():
            score += 10

        # UniProt reviewed entries get bonus points
        if record.get("source") == "UniProt":
            score += 10

        return min(score, 100.0)  # Cap at 100

    def comprehensive_gene_search(self) -> Dict[str, List[Dict]]:
        """
        Perform comprehensive search for all target genes across databases.

        Returns:
            Dictionary mapping gene names to lists of candidate records
        """
        logger.info(
            "Starting comprehensive gene search for perchlorate detoxification pathway"
        )

        all_results = {}

        for gene_code, gene_description in self.target_genes.items():
            logger.info(f"\n=== Searching for {gene_code} ({gene_description}) ===")

            gene_results = []

            # Search NCBI for each target organism
            for organism in self.target_organisms:
                logger.info(f"Searching NCBI for {gene_code} in {organism}")
                ncbi_results = self.search_ncbi_nucleotide(
                    gene_description, organism, max_results=10
                )
                gene_results.extend(ncbi_results)
                time.sleep(0.5)  # Be nice to NCBI servers

            # Search UniProt
            logger.info(f"Searching UniProt for {gene_code}")
            uniprot_results = self.search_uniprot(gene_description, max_results=20)
            gene_results.extend(uniprot_results)

            # Also search with gene code
            if gene_code != gene_description:
                uniprot_code_results = self.search_uniprot(gene_code, max_results=10)
                gene_results.extend(uniprot_code_results)

            # Score and rank results
            for result in gene_results:
                result["suitability_score"] = self.score_gene_suitability(result)

            # Sort by suitability score
            gene_results.sort(key=lambda x: x["suitability_score"], reverse=True)

            # Remove duplicates based on sequence similarity (basic check)
            unique_results = []
            seen_sequences = set()

            for result in gene_results:
                seq = result.get("sequence", "")
                if seq and seq not in seen_sequences:
                    seen_sequences.add(seq)
                    unique_results.append(result)
                elif not seq:  # Keep entries without sequences for reference
                    unique_results.append(result)

            all_results[gene_code] = unique_results[:15]  # Keep top 15 candidates

            logger.info(
                f"Found {len(unique_results)} unique candidates for {gene_code}"
            )

        return all_results

    def save_gene_sequences(self, search_results: Dict[str, List[Dict]]) -> None:
        """
        Save gene sequences to FASTA files and create summary reports.

        Args:
            search_results: Dictionary of search results from comprehensive_gene_search
        """
        logger.info("Saving gene sequences and creating summary reports")

        for gene_code, candidates in search_results.items():
            # Save FASTA sequences
            fasta_file = self.results_dir / f"{gene_code}_candidates.fasta"
            csv_file = self.results_dir / f"{gene_code}_summary.csv"

            sequences = []
            summary_data = []

            for i, candidate in enumerate(candidates):
                if candidate.get("sequence"):
                    # Create SeqRecord for FASTA
                    seq_id = f"{gene_code}_{i+1:02d}_{candidate.get('id', 'unknown')}"
                    description = f"{candidate.get('organism', 'Unknown')} | {candidate.get('definition', '')} | Score: {candidate.get('suitability_score', 0):.1f}"

                    seq_record = SeqRecord(
                        Seq(candidate["sequence"]), id=seq_id, description=description
                    )
                    sequences.append(seq_record)

                # Add to summary data
                summary_data.append(
                    {
                        "rank": i + 1,
                        "id": candidate.get("id", ""),
                        "organism": candidate.get("organism", ""),
                        "gene_symbol": candidate.get("gene_symbol", ""),
                        "length": candidate.get("length", 0),
                        "suitability_score": candidate.get("suitability_score", 0),
                        "source": candidate.get("source", ""),
                        "definition": candidate.get("definition", ""),
                        "function": candidate.get("function", ""),
                        "has_sequence": bool(candidate.get("sequence")),
                    }
                )

            # Write FASTA file
            if sequences:
                with open(fasta_file, "w") as f:
                    SeqIO.write(sequences, f, "fasta")
                logger.info(f"Saved {len(sequences)} sequences to {fasta_file}")

            # Write CSV summary
            df = pd.DataFrame(summary_data)
            df.to_csv(csv_file, index=False)
            logger.info(f"Saved summary to {csv_file}")

        # Create overall summary
        self._create_overall_summary(search_results)

    def _create_overall_summary(self, search_results: Dict[str, List[Dict]]) -> None:
        """Create an overall summary report of the gene search results."""

        summary_file = self.results_dir / "gene_search_summary.json"

        summary = {
            "search_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "target_genes": self.target_genes,
            "target_organisms": self.target_organisms,
            "results_summary": {},
        }

        for gene_code, candidates in search_results.items():
            gene_summary = {
                "total_candidates": len(candidates),
                "candidates_with_sequences": len(
                    [c for c in candidates if c.get("sequence")]
                ),
                "top_organism": candidates[0].get("organism", "") if candidates else "",
                "top_score": (
                    candidates[0].get("suitability_score", 0) if candidates else 0
                ),
                "source_distribution": {},
            }

            # Count sources
            for candidate in candidates:
                source = candidate.get("source", "unknown")
                gene_summary["source_distribution"][source] = (
                    gene_summary["source_distribution"].get(source, 0) + 1
                )

            summary["results_summary"][gene_code] = gene_summary

        with open(summary_file, "w") as f:
            json.dump(summary, f, indent=2)

        logger.info(f"Created overall summary at {summary_file}")


def main():
    """Main function to run comprehensive gene search."""

    # Initialize search engine
    search_engine = GeneSearchEngine(email="researcher@reddust-reclaimer.org")

    # Perform comprehensive search
    logger.info("Starting RedDust Reclaimer gene identification pipeline")
    results = search_engine.comprehensive_gene_search()

    # Save results
    search_engine.save_gene_sequences(results)

    # Print summary
    logger.info("\n=== GENE SEARCH SUMMARY ===")
    for gene_code, candidates in results.items():
        logger.info(f"{gene_code}: {len(candidates)} candidates found")
        if candidates:
            top_candidate = candidates[0]
            logger.info(
                f"  Top candidate: {top_candidate.get('organism', 'Unknown')} "
                f"(Score: {top_candidate.get('suitability_score', 0):.1f})"
            )

    logger.info("\nGene identification phase completed successfully!")


if __name__ == "__main__":
    main()
