#!/usr/bin/env python3
"""
Script de comparaison du graphe de connaissances AVANT/APRÈS Entity Resolution.

Usage:
    1. AVANT déploiement: python compare_graph_before_after.py --capture before --workspace USER_WORKSPACE
    2. Déployer la nouvelle version
    3. Supprimer et réindexer les documents
    4. APRÈS réindexation: python compare_graph_before_after.py --capture after --workspace USER_WORKSPACE
    5. Comparer: python compare_graph_before_after.py --compare --workspace USER_WORKSPACE
"""

import argparse
import asyncio
import json
import os
from datetime import datetime
from pathlib import Path


async def capture_graph_snapshot(workspace: str, output_dir: Path, label: str):
    """Capture l'état actuel du graphe."""
    from lightrag import LightRAG

    print(f"[{label.upper()}] Connexion au workspace: {workspace}")

    # Initialiser LightRAG avec le workspace
    rag = LightRAG(
        working_dir=os.getenv("LIGHTRAG_WORKING_DIR", "./rag_storage"),
        workspace=workspace,
    )

    await rag.initialize_storages()

    try:
        # Récupérer tous les nœuds (entités)
        print(f"[{label.upper()}] Récupération des entités...")
        nodes = await rag.chunk_entity_relation_graph.get_all_nodes()

        # Récupérer toutes les arêtes (relations)
        print(f"[{label.upper()}] Récupération des relations...")
        edges = await rag.chunk_entity_relation_graph.get_all_edges()

        # Sauvegarder le snapshot
        snapshot = {
            "timestamp": datetime.now().isoformat(),
            "workspace": workspace,
            "label": label,
            "stats": {
                "total_entities": len(nodes),
                "total_relations": len(edges),
                "entity_types": {},
            },
            "entities": nodes,
            "relations": edges,
        }

        # Compter par type d'entité
        for node in nodes:
            entity_type = node.get("entity_type", "UNKNOWN")
            snapshot["stats"]["entity_types"][entity_type] = (
                snapshot["stats"]["entity_types"].get(entity_type, 0) + 1
            )

        # Sauvegarder
        output_file = output_dir / f"graph_snapshot_{label}_{workspace}.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(snapshot, f, ensure_ascii=False, indent=2)

        print(f"\n[{label.upper()}] Snapshot sauvegardé: {output_file}")
        print(f"  - Entités: {len(nodes)}")
        print(f"  - Relations: {len(edges)}")
        print(f"  - Types d'entités: {snapshot['stats']['entity_types']}")

        return snapshot

    finally:
        await rag.finalize_storages()


def compare_snapshots(output_dir: Path, workspace: str):
    """Compare les snapshots BEFORE et AFTER."""
    before_file = output_dir / f"graph_snapshot_before_{workspace}.json"
    after_file = output_dir / f"graph_snapshot_after_{workspace}.json"

    if not before_file.exists():
        print(f"ERREUR: Snapshot BEFORE non trouvé: {before_file}")
        return

    if not after_file.exists():
        print(f"ERREUR: Snapshot AFTER non trouvé: {after_file}")
        return

    with open(before_file, "r", encoding="utf-8") as f:
        before = json.load(f)

    with open(after_file, "r", encoding="utf-8") as f:
        after = json.load(f)

    print("\n" + "=" * 70)
    print("COMPARAISON DU GRAPHE DE CONNAISSANCES")
    print("=" * 70)

    print(f"\nWorkspace: {workspace}")
    print(f"BEFORE: {before['timestamp']}")
    print(f"AFTER:  {after['timestamp']}")

    # Stats globales
    print("\n--- STATISTIQUES GLOBALES ---")
    print(f"{'Métrique':<30} {'BEFORE':>10} {'AFTER':>10} {'DIFF':>10}")
    print("-" * 60)

    before_entities = before["stats"]["total_entities"]
    after_entities = after["stats"]["total_entities"]
    diff_entities = after_entities - before_entities

    before_relations = before["stats"]["total_relations"]
    after_relations = after["stats"]["total_relations"]
    diff_relations = after_relations - before_relations

    print(f"{'Entités':<30} {before_entities:>10} {after_entities:>10} {diff_entities:>+10}")
    print(f"{'Relations':<30} {before_relations:>10} {after_relations:>10} {diff_relations:>+10}")

    # Réduction attendue avec Entity Resolution
    if diff_entities < 0:
        reduction_pct = abs(diff_entities) / before_entities * 100
        print(f"\n✅ RÉDUCTION D'ENTITÉS: {abs(diff_entities)} (-{reduction_pct:.1f}%)")
        print("   → Entity Resolution a fusionné des entités similaires")
    elif diff_entities == 0:
        print("\n⚠️  PAS DE CHANGEMENT dans le nombre d'entités")
    else:
        print(f"\n⚠️  AUGMENTATION D'ENTITÉS: {diff_entities}")

    # Comparaison par type d'entité
    print("\n--- PAR TYPE D'ENTITÉ ---")
    all_types = set(before["stats"]["entity_types"].keys()) | set(
        after["stats"]["entity_types"].keys()
    )

    print(f"{'Type':<25} {'BEFORE':>10} {'AFTER':>10} {'DIFF':>10}")
    print("-" * 55)

    for entity_type in sorted(all_types):
        b_count = before["stats"]["entity_types"].get(entity_type, 0)
        a_count = after["stats"]["entity_types"].get(entity_type, 0)
        diff = a_count - b_count
        print(f"{entity_type:<25} {b_count:>10} {a_count:>10} {diff:>+10}")

    # Identifier les entités fusionnées
    print("\n--- ANALYSE DES FUSIONS ---")

    before_names = {n.get("entity_name", n.get("id", "")): n for n in before["entities"]}
    after_names = {n.get("entity_name", n.get("id", "")): n for n in after["entities"]}

    # Entités présentes BEFORE mais pas AFTER (potentiellement fusionnées)
    removed = set(before_names.keys()) - set(after_names.keys())
    added = set(after_names.keys()) - set(before_names.keys())

    if removed:
        print(f"\nEntités SUPPRIMÉES/FUSIONNÉES ({len(removed)}):")
        for name in sorted(list(removed)[:20]):  # Limiter à 20
            print(f"  - {name}")
        if len(removed) > 20:
            print(f"  ... et {len(removed) - 20} autres")

    if added:
        print(f"\nNouvelles entités ({len(added)}):")
        for name in sorted(list(added)[:20]):
            print(f"  + {name}")
        if len(added) > 20:
            print(f"  ... et {len(added) - 20} autres")

    # Sauvegarder le rapport
    report_file = output_dir / f"comparison_report_{workspace}.txt"
    with open(report_file, "w", encoding="utf-8") as f:
        f.write(f"Comparaison du graphe - {workspace}\n")
        f.write(f"BEFORE: {before_entities} entités, {before_relations} relations\n")
        f.write(f"AFTER:  {after_entities} entités, {after_relations} relations\n")
        f.write(f"DIFF:   {diff_entities:+} entités, {diff_relations:+} relations\n")
        f.write(f"\nEntités supprimées: {len(removed)}\n")
        for name in sorted(removed):
            f.write(f"  - {name}\n")

    print(f"\n📄 Rapport sauvegardé: {report_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Comparer le graphe de connaissances AVANT/APRÈS Entity Resolution"
    )
    parser.add_argument(
        "--capture",
        choices=["before", "after"],
        help="Capturer un snapshot (before ou after)",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Comparer les snapshots before et after",
    )
    parser.add_argument(
        "--workspace",
        required=True,
        help="Workspace du user à analyser",
    )
    parser.add_argument(
        "--output-dir",
        default="./graph_comparison",
        help="Répertoire pour les snapshots (défaut: ./graph_comparison)",
    )

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.capture:
        asyncio.run(capture_graph_snapshot(args.workspace, output_dir, args.capture))
    elif args.compare:
        compare_snapshots(output_dir, args.workspace)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
