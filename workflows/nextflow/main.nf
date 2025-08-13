#!/usr/bin/env nextflow

/*
 * RedDust Reclaimer - Nextflow master workflow
 * Stages: gene_opt -> pathway_model -> enzyme_eng -> validation
 * Uses repo scripts under scripts/ and publishes artifacts to results/ by default
 */

nextflow.enable.dsl=2

params.sequence      = params.sequence ?: 'ATGAAATTTGGGTAG'
params.model         = params.model ?: 'models/mars_microbe_core.xml'
params.outdir        = params.outdir ?: 'results/nextflow'
params.threads       = params.threads ?: 2
params.dry_run       = params.dry_run ?: false

// Where to publish results
publishDir = params.outdir

// Allow using conda for reproducibility (requires Nextflow with conda support)
process.conda = 'environment.yml'
process.executor = 'local'
process.errorStrategy = 'retry'
process.maxRetries = 1
process.cpus = { params.threads as int }

// Input channels
Channel
    .of(params.sequence)
    .set { ch_seq }

Channel
    .of(params.model)
    .set { ch_model }

// 1) Gene optimization
process GENE_OPTIMIZATION {
    tag "codon_opt"
    publishDir publishDir, mode: 'copy', overwrite: true
    conda true

    input:
    val seq from ch_seq

    output:
    path 'codon_optimized.fasta' into ch_opt_seq
    path 'codon_opt_report.json' into ch_opt_report

    script:
    def runFlag = params.dry_run ? '--dry-run' : ''
    def cmd = """
    python scripts/codon_optimization.py \
      --sequence ${seq} \
      --output codon_optimized.fasta \
      --report codon_opt_report.json \
      --analyze ${runFlag}
    """
    return cmd
}

// 2) Metabolic pathway modeling (FBA)
process PATHWAY_MODELING {
    tag "metabolic_flux"
    publishDir publishDir, mode: 'copy', overwrite: true
    conda true

    input:
    path opt_seq from ch_opt_seq
    val model_path from ch_model

    output:
    path 'flux_summary.json' into ch_flux_summary
    path 'flux_plot.png' into ch_flux_plot

    script:
    def runFlag = params.dry_run ? '--dry-run' : ''
    def cmd = """
    python scripts/metabolic_flux.py \
      --model ${model_path} \
      --objective biomass_synthesis \
      --plot \
      --output flux_summary.json ${runFlag}
    """
    return cmd
}

// 3) Enzyme docking demo (placeholder for enzyme engineering)
process ENZYME_ENGINEERING {
    tag "docking_demo"
    publishDir publishDir, mode: 'copy', overwrite: true
    conda true

    input:
    path flux_summary from ch_flux_summary

    output:
    path 'docking_results_demo.txt' into ch_dock_results

    script:
    def cmd = """
    python scripts/dock_example.py --help > docking_results_demo.txt || true
    """
    return cmd
}

// 4) SBML model validation
process VALIDATION {
    tag "validate_models"
    publishDir publishDir, mode: 'copy', overwrite: true
    conda true

    input:
    val model_path from ch_model

    output:
    path 'validation.log' into ch_validation

    script:
    def cmd = """
    python scripts/validate_models.py > validation.log 2>&1 || true
    """
    return cmd
}

// Workflow definition
workflow {
    take: ch_seq, ch_model
    main:
        GENE_OPTIMIZATION(ch_seq)
        PATHWAY_MODELING(GENE_OPTIMIZATION.out[0], ch_model)
        ENZYME_ENGINEERING(PATHWAY_MODELING.out[0])
        VALIDATION(ch_model)
    emit:
        ENZYME_ENGINEERING.out[0], VALIDATION.out[0]
}
