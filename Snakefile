####################################################################
#   When the annotation changes, these variables need to be updated
####################################################################
ANNOTATION_F = '/ua/mnbernstein/projects/tbcp/phenotyping/manage_data/annotations/from.cell_type_primary_cells_v1-4/cell_type_primary_cells_v1-4.annot_5-1.experiment_centric.json'
ENV_NAME = 'cell_type.metasra_1-4.annot_5-1'

ENV_DIR = '/tier2/deweylab/mnbernstein/phenotyping_environments/{}'.format(ENV_NAME)
PROJECT_DIR = '/ua/mnbernstein/projects/tbcp/phenotyping'
TMP_DIR = '/scratch/mnbernstein/cell_type_phenotyping_tmp'
EXPERIMENT_LISTS = [
    'all_untampered_bulk_primary_cells_with_data',
    'untampered_bulk_primary_cells_with_data',
    'train_set.untampered_bulk_primary_cells_with_data',
    'test_set.untampered_bulk_primary_cells_with_data',
    'untampered_single_cell_primary_cells_with_data',
    'test_experiments',
    'toy',
    'toy_single_cell' 
]
EXPERIMENT_LISTS_2 = [
    'untampered_single_cell_primary_cells_with_data_cell_types_in_bulk'
]

EXPERIMENT_LISTS_3 = [
    'untampered_cells_with_data'
]

ONE_NN_LOG_CPM_CORRELATION = "one_nn.correlation.kallisto_gene_log_cpm"
CDC_LOG_CPM_LOGISTIC_L2_ASSERT_AMBIG_NEG_COLLAPSE = "cdc.logistic_regression.l2.assert_ambig_neg.collapse_ontology"
CDC_LOG_CPM_LOGISTIC_L2_ASSERT_AMBIG_NEG_FULL = "cdc.logistic_regression.l2.assert_ambig_neg.full_ontology"

CDC_LOG_CPM_LOGISTIC_L2_REMOVE_AMBIG_COLLAPSE = "cdc.logistic_regression.l2.remove_ambig.collapse_ontology"
CDC_LOG_CPM_LOGISTIC_L2_REMOVE_AMBIG_FULL = "cdc.logistic_regression.l2.remove_ambig.full_ontology"
CDC_LOG_CPM_LOGISTIC_L2_ASSERT_AMBIG_NEG_COLLAPSE_DOWNWEIGHT = "cdc.logistic_regression.l2.assert_ambig_neg.downweight_by_group.collapse_ontology"
CDC_LOG_CPM_LOGISTIC_L2_ASSERT_AMBIG_NEG_FULL_DOWNWEIGHT = "cdc.logistic_regression.l2.assert_ambig_neg.downweight_by_group.full_ontology"

ISOTONIC_LOGISTIC_L2_LOG_CPM_FULL = "isotonic.logistic_regression.l2.full_ontology"
ISOTONIC_LOGISTIC_L2_LOG_CPM_ASSERT_AMBIG_NEG_FULL = "isotonic.logistic_regression.l2.assert_ambig_neg.full_ontology"
ISOTONIC_LOGISTIC_L2_LOG_CPM_FULL_DOWNWEIGHT = "isotonic.logistic_regression.l2.downweight_by_group.full_ontology"
ISOTONIC_LOGISTIC_L2_LOG_CPM_ASSERT_AMBIG_NEG_FULL_DOWNWEIGHT = "isotonic.logistic_regression.l2.downweight_by_group.assert_ambig_neg.full_ontology"

BNC_LOG_CPM_LINEAR_SVM = "bnc.linear_svm.dynamic_bins_10.bin_pseudo_1.counts_prior.gibbs.burn_in_30000"
BNC_LOG_CPM_LINEAR_SVM_CONSTANT_PRIOR = "bnc.linear_svm.dynamic_bins_10.bin_pseudo_1.constant.gibbs.burn_in_30000"
BNC_LOG_CPM_LINEAR_SVM_NORMAL = "bnc.linear_svm.normal.conditional_counts_prior.gibbs.burn_in_30000"
BNC_LOG_CPM_LINEAR_SVM_NORMAL_FIXED_STD = "bnc.linear_svm.normal.fixed_std.conditional_counts_prior.gibbs.burn_in_30000"

BNC_LOG_CPM_LOGISTIC_L2 = "bnc.logistic.l2.dynamic_bins_10.bin_pseudo_1.counts_prior.gibbs.burn_in_30000"
BNC_LOG_CPM_LOGISTIC_L2_PSEUDO_0_1 = "bnc.logistic.l2.dynamic_bins_10.bin_pseudo_0_1.counts_prior.gibbs.burn_in_30000"
BNC_LOG_CPM_LOGISTIC_L2_DOWNWEIGHT = "bnc.logistic.l2.downweight_by_group.dynamic_bins_10.bin_pseudo_1.counts_prior.gibbs.burn_in_30000"
BNC_LOG_CPM_LOGISTIC_L2_CONSTANT_PRIOR = "bnc.logistic.l2.dynamic_bins_10.bin_pseudo_1.constant.gibbs.burn_in_30000"
BNC_LOG_CPM_LOGISTIC_L2_COND_DOWNDWIEGHT = "bnc.logistic.l2.downweight_by_group.dynamic_bins_10.bin_pseudo_1.conditional_counts_prior.gibbs.burn_in_30000"

PER_LABEL_LOG_CPM_LOGISTIC_L2_FULL = "per_label.logistic_regression.l2.full_ontology"
PER_LABEL_LOG_CPM_LOGISTIC_L2_FULL_DOWNWEIGHT = "per_label.logistic_regression.l2.downweight_by_group.full_ontology"

BNC_LOG_CPM_LOGISTIC_L2_COND = "bnc.logistic.l2.dynamic_bins_10.bin_pseudo_1.conditional_counts_prior.gibbs.burn_in_30000"
BNC_LOG_CPM_LOGISTIC_L2_COND_STATIC_BINS = "bnc.logistic.l2.static_bins_10.bin_pseudo_1.conditional_counts_prior.gibbs.burn_in_30000"
BNc_LOG_CPM_LOGISTIC_L2_COND_NO_ASSERT_AMBIG_NEG = "bnc.logistic.l2.dynamic_bins_10.bin_pseudo_1.conditional_counts_prior.gibbs.burn_in_30000.no_assert_ambig_neg"

BNC_LOG_CPM_LINEAR_SVM_COND = "bnc.linear_svm.dynamic_bins_10.bin_pseudo_1.conditional_counts_prior.gibbs.burn_in_30000"
BNC_LOG_CPM_LINEAR_SVM_COND_ASSERT_AMBIG_NEG = "bnc.linear_svm.dynamic_bins_10.bin_pseudo_1.conditional_counts_prior.gibbs.burn_in_30000.assert_ambig_neg"


NAIVE_BAYES_SVM_COND_ASSERT_AMBIG_NEG = "naive_bayes.linear_svm.dynamic_bins_10.assert_ambig_neg"
NAIVE_BAYES_SVM_COUNTS_ASSERT_AMBIG_NEG = "naive_bayes.linear_svm.dynamic_bins_10.counts_prior.assert_ambig_neg"


TPR_LOGISTIC_L2_LOG_CPM_FULL = "true_path_rule.logistic_regression.l2.assert_ambig_neg.full_ontology"
TPR_LOGISTIC_L2_LOG_CPM_FULL_DOWNWEIGHT = "true_path_rule.logistic_regression.l2.assert_ambig_neg.downweight_by_group.full_ontology"

rule all:
    input:
        expand('{}/data/experiment_lists/{{exp_list}}/experiment_list.json'.format(ENV_DIR), exp_list=EXPERIMENT_LISTS),
        expand('{}/data/experiment_lists/{{exp_list}}/labelling.json'.format(ENV_DIR), exp_list=EXPERIMENT_LISTS),
        expand('{}/data/experiment_lists/{{exp_list}}/labelling_collapsed_dag.json'.format(ENV_DIR), exp_list=EXPERIMENT_LISTS)
        # TODO add all outputs?
        
rule visualizations:
    input:
        expand('{}/data/experiment_lists/{{exp_list}}/summary_figures/pca_color_by_child_on_graph.pdf'.format(ENV_DIR), exp_list=EXPERIMENT_LISTS),
        expand('{}/data/experiment_lists/{{exp_list}}/summary_figures/num_genes_expressed_distribution.pdf'.format(ENV_DIR), exp_list=EXPERIMENT_LISTS)

rule setup:
    input:
        '{}/data/data_set_metadata.json'.format(ENV_DIR),
        expand('{}/data/experiment_lists/{{exp_list}}/experiment_list.json'.format(ENV_DIR), exp_list=EXPERIMENT_LISTS),
        expand('{}/data/experiment_lists/{{exp_list}}/experiment_list.json'.format(ENV_DIR), exp_list=EXPERIMENT_LISTS_2),
        expand('{}/data/experiment_lists/{{exp_list}}/labelling.json'.format(ENV_DIR), exp_list=EXPERIMENT_LISTS),
        expand('{}/data/experiment_lists/{{exp_list}}/experiment_list.json'.format(ENV_DIR), exp_list=EXPERIMENT_LISTS_3),
        expand('{}/data/experiment_lists/{{exp_list}}/labelling.json'.format(ENV_DIR), exp_list=EXPERIMENT_LISTS_2),
        expand('{}/data/experiment_lists/{{exp_list}}/labelling.json'.format(ENV_DIR), exp_list=EXPERIMENT_LISTS_3),
        expand('{}/data/experiment_lists/{{exp_list}}/labelling_collapsed_dag.json'.format(ENV_DIR), exp_list=EXPERIMENT_LISTS),
        '{env_dir}/data/experiment_lists/{exp_list}/trained_models/{algo_config}/model.pickle'.format(
            env_dir=ENV_DIR,
            exp_list='untampered_bulk_primary_cells_with_data',
            algo_config=CDC_LOG_CPM_LOGISTIC_L2_ASSERT_AMBIG_NEG_FULL_DOWNWEIGHT
        ),
        '{env_dir}/data/experiment_lists/{exp_list}/trained_on_another_experiment_list/{train_exp_list}/{algo_config}/prediction_results.json'.format(
            env_dir=ENV_DIR,
            exp_list='untampered_single_cell_primary_cells_with_data_cell_types_in_bulk',
            train_exp_list='untampered_bulk_primary_cells_with_data',
            algo_config=CDC_LOG_CPM_LOGISTIC_L2_ASSERT_AMBIG_NEG_FULL_DOWNWEIGHT
        ),
        '{env_dir}/data/experiment_lists/{exp_list}/trained_models/{algo_config}/model.pickle'.format(
            env_dir=ENV_DIR,
            exp_list='train_set.untampered_bulk_primary_cells_with_data',
            algo_config=ISOTONIC_LOGISTIC_L2_LOG_CPM_ASSERT_AMBIG_NEG_FULL_DOWNWEIGHT
        ),
        '{env_dir}/data/experiment_lists/{exp_list}/trained_models/{algo_config}/model.pickle'.format(
            env_dir=ENV_DIR,
            exp_list='untampered_bulk_primary_cells_with_data',
            algo_config=TPR_LOGISTIC_L2_LOG_CPM_FULL_DOWNWEIGHT
        )


####################################################################
#   Perform the experiment that analyzes the effect of training
#   on heterogenous versus homogenous data
####################################################################
rule effects_of_heterogeneous_data_analysis:
    input:
        '{}/data/experiment_lists/untampered_bulk_primary_cells_with_data/experiment_list.json'.format(ENV_DIR),
        '{}/data/experiment_lists/untampered_bulk_primary_cells_with_data/labelling.json'.format(ENV_DIR)
    output:
        '{}/data/experiment_lists/untampered_bulk_primary_cells_with_data/analyze_effects_of_heterogeneous_data/homo_vs_hetero_training_results.min_10.json'.format(ENV_DIR)
    run:
        commands = [
            'mkdir -p {}/data/experiment_lists/untampered_bulk_primary_cells_with_data/analyze_effects_of_heterogeneous_data'.format(ENV_DIR),
            'python2.7 {project_dir}/learning/analyze_effects_of_heterogeneous_data/analyze_effects_of_heterogeneous_data.py {env_dir} untampered_bulk_primary_cells_with_data -o {env_dir}/data/experiment_lists/untampered_bulk_primary_cells_with_data/analyze_effects_of_heterogeneous_data'.format(
                project_dir=PROJECT_DIR,
                env_dir=ENV_DIR
            )
        ]
        for c in commands:
            shell(c)


####################################################################
#   Perform GO gene set enrichment analysis on coefficients
####################################################################
rule gene_set_enrichment_analysis_cdc:
    input:
        model = '{env_dir}/data/experiment_lists/untampered_bulk_primary_cells_with_data/trained_models/{algo_config}/model.pickle'.format(
            env_dir=ENV_DIR,
            algo_config=CDC_LOG_CPM_LOGISTIC_L2_ASSERT_AMBIG_NEG_FULL
        )
    output:
        '{env_dir}/data/experiment_lists/untampered_bulk_primary_cells_with_data/gsea_results/{algo_config}/coefficient_enriched_GO_terms.json'.format(
            env_dir=ENV_DIR,
            algo_config=CDC_LOG_CPM_LOGISTIC_L2_ASSERT_AMBIG_NEG_FULL
        )
    run:
        commands = [
            'mkdir -p {env_dir}/data/experiment_lists/untampered_bulk_primary_cells_with_data/gsea_results/{algo_config}'.format(
                env_dir=ENV_DIR,
                algo_config=CDC_LOG_CPM_LOGISTIC_L2_ASSERT_AMBIG_NEG_FULL
            ),
            'mkdir -p /scratch/mnbernstein/gsea_coeffs_condor/untampered_bulk_primary_cells_with_data.{}'.format(
                CDC_LOG_CPM_LOGISTIC_L2_ASSERT_AMBIG_NEG_FULL
            ),
            'python2.7 {project_dir}/learning/interpret_models/gsea_coeffs_condor_primary.py {env_dir} untampered_bulk_primary_cells_with_data {{input.model}} -c /scratch/mnbernstein/gsea_coeffs_condor/untampered_bulk_primary_cells_with_data.{algo_config} -o {{output}}'.format(
                project_dir=PROJECT_DIR,
                env_dir=ENV_DIR,
                algo_config=CDC_LOG_CPM_LOGISTIC_L2_ASSERT_AMBIG_NEG_FULL
            )
        ]
        for c in commands:
            shell(c)


rule extract_GO_terms_cdc:
    input:
        '{env_dir}/data/experiment_lists/untampered_bulk_primary_cells_with_data/gsea_results/{algo_config}/coefficient_enriched_GO_terms.json'.format(
            env_dir=ENV_DIR,
            algo_config=CDC_LOG_CPM_LOGISTIC_L2_ASSERT_AMBIG_NEG_FULL
        )
    output:
        '{env_dir}/data/experiment_lists/untampered_bulk_primary_cells_with_data/gsea_results/{algo_config}/enriched_GO_terms_summary.json'.format(
            env_dir=ENV_DIR,
            algo_config=CDC_LOG_CPM_LOGISTIC_L2_ASSERT_AMBIG_NEG_FULL
        )
    run:
        commands = [
            'python2.7 {}/learning/interpret_models/extract_statistically_enriched_terms.py {{input}} -o {{output}}'.format(PROJECT_DIR)
        ]
        for c in commands:
            shell(c)


rule gene_set_enrichment_analysis_cdc_downweight:
    input:
        model = '{env_dir}/data/experiment_lists/untampered_bulk_primary_cells_with_data/trained_models/{algo_config}/model.pickle'.format(
            env_dir=ENV_DIR,
            algo_config=CDC_LOG_CPM_LOGISTIC_L2_ASSERT_AMBIG_NEG_FULL_DOWNWEIGHT
        )
    output:
        '{env_dir}/data/experiment_lists/untampered_bulk_primary_cells_with_data/gsea_results/{algo_config}/coefficient_enriched_GO_terms.json'.format(
            env_dir=ENV_DIR,
            algo_config=CDC_LOG_CPM_LOGISTIC_L2_ASSERT_AMBIG_NEG_FULL_DOWNWEIGHT
        )
    run:
        commands = [
            'mkdir -p {env_dir}/data/experiment_lists/untampered_bulk_primary_cells_with_data/gsea_results/{algo_config}'.format(
                env_dir=ENV_DIR,
                algo_config=CDC_LOG_CPM_LOGISTIC_L2_ASSERT_AMBIG_NEG_FULL_DOWNWEIGHT
            ),
            'mkdir -p /scratch/mnbernstein/gsea_coeffs_condor/untampered_bulk_primary_cells_with_data.{}'.format(
                CDC_LOG_CPM_LOGISTIC_L2_ASSERT_AMBIG_NEG_FULL_DOWNWEIGHT
            ),
            'python2.7 {project_dir}/learning/interpret_models/gsea_coeffs_condor_primary.py {env_dir} untampered_bulk_primary_cells_with_data {{input.model}} -c /scratch/mnbernstein/gsea_coeffs_condor/untampered_bulk_primary_cells_with_data.{algo_config} -o {{output}}'.format(
                project_dir=PROJECT_DIR,
                env_dir=ENV_DIR,
                algo_config=CDC_LOG_CPM_LOGISTIC_L2_ASSERT_AMBIG_NEG_FULL_DOWNWEIGHT
            )
        ]
        for c in commands:
            shell(c)

rule extract_GO_terms_cdc_downweight:
    input:
        '{env_dir}/data/experiment_lists/untampered_bulk_primary_cells_with_data/gsea_results/{algo_config}/coefficient_enriched_GO_terms.json'.format(
            env_dir=ENV_DIR,
            algo_config=CDC_LOG_CPM_LOGISTIC_L2_ASSERT_AMBIG_NEG_FULL_DOWNWEIGHT
        )
    output:
        '{env_dir}/data/experiment_lists/untampered_bulk_primary_cells_with_data/gsea_results/{algo_config}/enriched_GO_terms_summary.json'.format(
            env_dir=ENV_DIR,
            algo_config=CDC_LOG_CPM_LOGISTIC_L2_ASSERT_AMBIG_NEG_FULL_DOWNWEIGHT
        )
    run:
        commands = [
            'python2.7 {}/learning/interpret_models/extract_statistically_enriched_terms.py {{input}} -o {{output}}'.format(PROJECT_DIR)
        ]
        for c in commands:
            shell(c)

rule gene_set_enrichment_analysis_isotonic_regression:
    input:
        model = '{env_dir}/data/experiment_lists/untampered_bulk_primary_cells_with_data/trained_models/{algo_config}/model.pickle'.format(
            env_dir=ENV_DIR,
            algo_config=ISOTONIC_LOGISTIC_L2_LOG_CPM_ASSERT_AMBIG_NEG_FULL
        )
    output:
        '{env_dir}/data/experiment_lists/untampered_bulk_primary_cells_with_data/gsea_results/{algo_config}/coefficient_enriched_GO_terms.json'.format(
            env_dir=ENV_DIR,
            algo_config=ISOTONIC_LOGISTIC_L2_LOG_CPM_ASSERT_AMBIG_NEG_FULL
        )
    run:
        commands = [
            'mkdir -p {env_dir}/data/experiment_lists/untampered_bulk_primary_cells_with_data/gsea_results/{algo_config}'.format(
                env_dir=ENV_DIR,
                algo_config=ISOTONIC_LOGISTIC_L2_LOG_CPM_ASSERT_AMBIG_NEG_FULL
            ),
            'mkdir -p /scratch/mnbernstein/gsea_coeffs_condor/untampered_bulk_primary_cells_with_data.{}'.format(
                ISOTONIC_LOGISTIC_L2_LOG_CPM_ASSERT_AMBIG_NEG_FULL
            ),
            'python2.7 {project_dir}/learning/interpret_models/gsea_coeffs_condor_primary.py {env_dir} untampered_bulk_primary_cells_with_data {{input.model}} -c /scratch/mnbernstein/gsea_coeffs_condor/untampered_bulk_primary_cells_with_data.{algo_config} -o {{output}}'.format(
                project_dir=PROJECT_DIR,
                env_dir=ENV_DIR,
                algo_config=ISOTONIC_LOGISTIC_L2_LOG_CPM_ASSERT_AMBIG_NEG_FULL
            )
        ]
        for c in commands:
            shell(c)

rule extract_GO_terms_isotonic:
    input:
        '{env_dir}/data/experiment_lists/untampered_bulk_primary_cells_with_data/gsea_results/{algo_config}/coefficient_enriched_GO_terms.json'.format(
            env_dir=ENV_DIR,
            algo_config=ISOTONIC_LOGISTIC_L2_LOG_CPM_ASSERT_AMBIG_NEG_FULL
        )
    output:
        '{env_dir}/data/experiment_lists/untampered_bulk_primary_cells_with_data/gsea_results/{algo_config}/enriched_GO_terms_summary.json'.format(
            env_dir=ENV_DIR,
            algo_config=ISOTONIC_LOGISTIC_L2_LOG_CPM_ASSERT_AMBIG_NEG_FULL
        )
    run:
        commands = [
            'python2.7 {}/learning/interpret_models/extract_statistically_enriched_terms.py {{input}} -o {{output}}'.format(PROJECT_DIR)
        ]
        for c in commands:
            shell(c)

rule gene_set_enrichment_analysis_isotonic_regression_downweight:
    input:
        model = '{env_dir}/data/experiment_lists/untampered_bulk_primary_cells_with_data/trained_models/{algo_config}/model.pickle'.format(
            env_dir=ENV_DIR,
            algo_config=ISOTONIC_LOGISTIC_L2_LOG_CPM_ASSERT_AMBIG_NEG_FULL_DOWNWEIGHT
        )
    output:
        '{env_dir}/data/experiment_lists/untampered_bulk_primary_cells_with_data/gsea_results/{algo_config}/coefficient_enriched_GO_terms.json'.format(
            env_dir=ENV_DIR,
            algo_config=ISOTONIC_LOGISTIC_L2_LOG_CPM_ASSERT_AMBIG_NEG_FULL_DOWNWEIGHT
        )
    run:
        commands = [
            'mkdir -p {env_dir}/data/experiment_lists/untampered_bulk_primary_cells_with_data/gsea_results/{algo_config}'.format(
                env_dir=ENV_DIR,
                algo_config=ISOTONIC_LOGISTIC_L2_LOG_CPM_ASSERT_AMBIG_NEG_FULL_DOWNWEIGHT
            ),
            'mkdir -p /scratch/mnbernstein/gsea_coeffs_condor/untampered_bulk_primary_cells_with_data.{}'.format(
                ISOTONIC_LOGISTIC_L2_LOG_CPM_ASSERT_AMBIG_NEG_FULL_DOWNWEIGHT
            ),
            'python2.7 {project_dir}/learning/interpret_models/gsea_coeffs_condor_primary.py {env_dir} untampered_bulk_primary_cells_with_data {{input.model}} -c /scratch/mnbernstein/gsea_coeffs_condor/untampered_bulk_primary_cells_with_data.{algo_config} -o {{output}}'.format(
                project_dir=PROJECT_DIR,
                env_dir=ENV_DIR,
                algo_config=ISOTONIC_LOGISTIC_L2_LOG_CPM_ASSERT_AMBIG_NEG_FULL_DOWNWEIGHT
            )
        ]
        for c in commands:
            shell(c)


rule extract_GO_terms_isotonic_downweight:
    input:
        '{env_dir}/data/experiment_lists/untampered_bulk_primary_cells_with_data/gsea_results/{algo_config}/coefficient_enriched_GO_terms.json'.format(
            env_dir=ENV_DIR,
            algo_config=ISOTONIC_LOGISTIC_L2_LOG_CPM_ASSERT_AMBIG_NEG_FULL_DOWNWEIGHT
        )
    output:
        '{env_dir}/data/experiment_lists/untampered_bulk_primary_cells_with_data/gsea_results/{algo_config}/enriched_GO_terms_summary.json'.format(
            env_dir=ENV_DIR,
            algo_config=ISOTONIC_LOGISTIC_L2_LOG_CPM_ASSERT_AMBIG_NEG_FULL_DOWNWEIGHT
        )
    run:
        commands = [
            'python2.7 {}/learning/interpret_models/extract_statistically_enriched_terms.py {{input}} -o {{output}}'.format(PROJECT_DIR)
        ]
        for c in commands:
            shell(c)

rule compare_GO_terms_isotonic_cdc_downweight:
    input:
        in_1='{env_dir}/data/experiment_lists/untampered_bulk_primary_cells_with_data/gsea_results/{algo_config}/enriched_GO_terms_summary.json'.format(
            env_dir=ENV_DIR,
            algo_config=ISOTONIC_LOGISTIC_L2_LOG_CPM_ASSERT_AMBIG_NEG_FULL_DOWNWEIGHT
        ),
        in_2='{env_dir}/data/experiment_lists/untampered_bulk_primary_cells_with_data/gsea_results/{algo_config}/enriched_GO_terms_summary.json'.format(
            env_dir=ENV_DIR,
            algo_config=CDC_LOG_CPM_LOGISTIC_L2_ASSERT_AMBIG_NEG_FULL_DOWNWEIGHT
        )
    output:
        '{env_dir}/data/experiment_lists/untampered_bulk_primary_cells_with_data/gsea_results/summary/{algo_config_1}-VS-{algo_config_2}/compare_number_of_enriched_GO_terms_boxplots.pdf'.format(
            env_dir=ENV_DIR,
            algo_config_1=ISOTONIC_LOGISTIC_L2_LOG_CPM_ASSERT_AMBIG_NEG_FULL_DOWNWEIGHT,
            algo_config_2=CDC_LOG_CPM_LOGISTIC_L2_ASSERT_AMBIG_NEG_FULL_DOWNWEIGHT
        ),
        '{env_dir}/data/experiment_lists/untampered_bulk_primary_cells_with_data/gsea_results/summary/{algo_config_1}-VS-{algo_config_2}/compare_number_of_enriched_GO_terms_scatterplot.pdf'.format(
            env_dir=ENV_DIR,
            algo_config_1=ISOTONIC_LOGISTIC_L2_LOG_CPM_ASSERT_AMBIG_NEG_FULL_DOWNWEIGHT,
            algo_config_2=CDC_LOG_CPM_LOGISTIC_L2_ASSERT_AMBIG_NEG_FULL_DOWNWEIGHT
        )
    run:
        commands = [
            'mkdir -p {env_dir}/data/experiment_lists/untampered_bulk_primary_cells_with_data/gsea_results/summary/{algo_config_1}-VS-{algo_config_2}'.format(
                env_dir=ENV_DIR,
                algo_config_1=ISOTONIC_LOGISTIC_L2_LOG_CPM_ASSERT_AMBIG_NEG_FULL_DOWNWEIGHT,
                algo_config_2=CDC_LOG_CPM_LOGISTIC_L2_ASSERT_AMBIG_NEG_FULL_DOWNWEIGHT
            ),
            "python2.7 {project_dir}/learning/interpret_models/compare_statistically_enriched_terms.py {{input.in_1}} {{input.in_2}} 'One-vs.-rest' 'Cascaded' -o {env_dir}/data/experiment_lists/untampered_bulk_primary_cells_with_data/gsea_results/summary/{algo_config_1}-VS-{algo_config_2}".format(
                project_dir=PROJECT_DIR,
                env_dir=ENV_DIR,
                algo_config_1=ISOTONIC_LOGISTIC_L2_LOG_CPM_ASSERT_AMBIG_NEG_FULL_DOWNWEIGHT,
                algo_config_2=CDC_LOG_CPM_LOGISTIC_L2_ASSERT_AMBIG_NEG_FULL_DOWNWEIGHT
            )
        ]
        for c in commands:
            shell(c)


#####################################################################
#   Run leave-study-out cross-fold validation on the full set
#   of bulk RNA-seq experiments. Analyze inconsistent edges using 
#   leave-study-out cross validation on entire training set
######################################################################

rule leave_study_out_cv_full_data_per_label_logistic_l2_condor:
    input:
        data_set='{}/data/data_set_metadata.json'.format(ENV_DIR),
        exp_list_f='{}/data/experiment_lists/untampered_bulk_primary_cells_with_data/experiment_list.json'.format(ENV_DIR),
        labelling_f='{}/data/experiment_lists/untampered_bulk_primary_cells_with_data/labelling.json'.format(ENV_DIR)
    output:
        '{env_dir}/data/experiment_lists/untampered_bulk_primary_cells_with_data/leave_study_out_cv_results/{cv_config}/leave_study_out_cv_results.json'.format(
            env_dir=ENV_DIR,
            cv_config=PER_LABEL_LOG_CPM_LOGISTIC_L2_FULL
        )
    run:
        commands = [
            'mkdir -p {tmp_dir}/leave_study_out_cv_condor/untampered_bulk_primary_cells_with_data/{cv_config}'.format(
                tmp_dir=TMP_DIR,
                cv_config=PER_LABEL_LOG_CPM_LOGISTIC_L2_FULL
            ),
            'mkdir -p {env_dir}/data/experiment_lists/untampered_bulk_primary_cells_with_data/leave_study_out_cv_results/{cv_config}'.format(
                env_dir=ENV_DIR,
                cv_config=PER_LABEL_LOG_CPM_LOGISTIC_L2_FULL
            ),
            'python2.7 {project_dir}/learning/leave_study_out_cv_condor_primary.py {project_dir}/learning/leave_study_out_cv_config/{env_name}/{cv_config}.json {env_dir} untampered_bulk_primary_cells_with_data -o {{output}} -r {tmp_dir}//leave_study_out_cv_condor/untampered_bulk_primary_cells_with_data/{cv_config}'.format(
                project_dir=PROJECT_DIR,
                env_name=ENV_NAME,
                env_dir=ENV_DIR,
                tmp_dir=TMP_DIR,
                cv_config=PER_LABEL_LOG_CPM_LOGISTIC_L2_FULL
            )
        ]
        for c in commands:
            shell(c)


rule analyze_inconsistent_edges:
    input:
        exp_list_f='{env_dir}/data/experiment_lists/{exp_list_name}/experiment_list.json'.format(
            env_dir=ENV_DIR,
            exp_list_name='untampered_bulk_primary_cells_with_data'
        ),
        labelling_f='{env_dir}/data/experiment_lists/{exp_list_name}/labelling.json'.format(
            env_dir=ENV_DIR,
            exp_list_name='untampered_bulk_primary_cells_with_data'
        ),
        results_f='{env_dir}/data/experiment_lists/{exp_list_name}/leave_study_out_cv_results/{algo_config}/leave_study_out_cv_results.json'.format(
            env_dir=ENV_DIR,
            exp_list_name='untampered_bulk_primary_cells_with_data',
            algo_config=PER_LABEL_LOG_CPM_LOGISTIC_L2_FULL
        )   
    output:
        '{env_dir}/data/experiment_lists/untampered_bulk_primary_cells_with_data/leave_study_out_cv_results/{algo_config}/summary/inconsistent_edges_stats.tsv'.format(
            env_dir=ENV_DIR,
            exp_list_name='untampered_bulk_primary_cells_with_data',
            algo_config=PER_LABEL_LOG_CPM_LOGISTIC_L2_FULL
        ),
        '{env_dir}/data/experiment_lists/untampered_bulk_primary_cells_with_data/leave_study_out_cv_results/{algo_config}/summary/CDF_inconsistences.eps'.format(
            env_dir=ENV_DIR,
            exp_list_name='untampered_bulk_primary_cells_with_data',
            algo_config=PER_LABEL_LOG_CPM_LOGISTIC_L2_FULL
        )
    run:
        commands = [
            'mkdir -p {env_dir}/data/experiment_lists/untampered_bulk_primary_cells_with_data/leave_study_out_cv_results/{algo_config}/summary'.format(
            env_dir=ENV_DIR,
            exp_list_name='untampered_bulk_primary_cells_with_data',
            algo_config=PER_LABEL_LOG_CPM_LOGISTIC_L2_FULL
            ),
            'python2.7 {project_dir}/learning/analyze_inconsistent_probabilities/analyze_inconsistent_labellings.py {env_dir} {exp_list_name} {{input.results_f}} -o {env_dir}/data/experiment_lists/untampered_bulk_primary_cells_with_data/leave_study_out_cv_results/{algo_config}/summary'.format(
                env_dir=ENV_DIR,
                project_dir=PROJECT_DIR,
                exp_list_name='untampered_bulk_primary_cells_with_data',
                algo_config=PER_LABEL_LOG_CPM_LOGISTIC_L2_FULL
            )
        ]
        for c in commands:
            shell(c)


#####################################################################
#   Run leave-study-out cross-fold validation on the training set
#   of experiments
#####################################################################

rule bnc_linear_svm_full_leave_study_out_cv_condor:
    input:
        data_set='{}/data/data_set_metadata.json'.format(ENV_DIR),
        exp_list_f='{}/data/experiment_lists/train_set.untampered_bulk_primary_cells_with_data/experiment_list.json'.format(ENV_DIR),
        labelling_f='{}/data/experiment_lists/train_set.untampered_bulk_primary_cells_with_data/labelling.json'.format(ENV_DIR)
    output:
        '{env_dir}/data/experiment_lists/train_set.untampered_bulk_primary_cells_with_data/leave_study_out_cv_results/{cv_config}/leave_study_out_cv_results.json'.format(
            env_dir=ENV_DIR,
            cv_config=BNC_LOG_CPM_LINEAR_SVM
        )
    run:
        commands = [
            'mkdir -p {tmp_dir}/leave_study_out_cv_condor/train_set.untampered_bulk_primary_cells_with_data/{cv_config}'.format(
                tmp_dir=TMP_DIR,
                cv_config=BNC_LOG_CPM_LINEAR_SVM
            ),
            'mkdir -p {env_dir}/data/experiment_lists/train_set.untampered_bulk_primary_cells_with_data/leave_study_out_cv_results/{cv_config}'.format(
                env_dir=ENV_DIR,
                cv_config=BNC_LOG_CPM_LINEAR_SVM
            ),
            'python2.7 {project_dir}/learning/leave_study_out_cv_condor_primary.py {project_dir}/learning/leave_study_out_cv_config/{env_name}/{cv_config}.json {env_dir} train_set.untampered_bulk_primary_cells_with_data -o {{output}} -r {tmp_dir}//leave_study_out_cv_condor/train_set.untampered_bulk_primary_cells_with_data/{cv_config}'.format(
                project_dir=PROJECT_DIR,
                env_name=ENV_NAME,
                env_dir=ENV_DIR,
                tmp_dir=TMP_DIR,
                cv_config=BNC_LOG_CPM_LINEAR_SVM
            )
        ]
        for c in commands:
            shell(c)

rule bnc_linear_svm_full_constant_leave_study_out_cv_condor:
    input:
        data_set='{}/data/data_set_metadata.json'.format(ENV_DIR),
        exp_list_f='{}/data/experiment_lists/train_set.untampered_bulk_primary_cells_with_data/experiment_list.json'.format(ENV_DIR),
        labelling_f='{}/data/experiment_lists/train_set.untampered_bulk_primary_cells_with_data/labelling.json'.format(ENV_DIR)
    output:
        '{env_dir}/data/experiment_lists/train_set.untampered_bulk_primary_cells_with_data/leave_study_out_cv_results/{cv_config}/leave_study_out_cv_results.json'.format(
            env_dir=ENV_DIR,
            cv_config=BNC_LOG_CPM_LINEAR_SVM_CONSTANT_PRIOR
        )
    run:
        commands = [
            'mkdir -p {tmp_dir}/leave_study_out_cv_condor/train_set.untampered_bulk_primary_cells_with_data/{cv_config}'.format(
                tmp_dir=TMP_DIR,
                cv_config=BNC_LOG_CPM_LINEAR_SVM_CONSTANT_PRIOR
            ),
            'mkdir -p {env_dir}/data/experiment_lists/train_set.untampered_bulk_primary_cells_with_data/leave_study_out_cv_results/{cv_config}'.format(
                env_dir=ENV_DIR,
                cv_config=BNC_LOG_CPM_LINEAR_SVM_CONSTANT_PRIOR
            ),
            'python2.7 {project_dir}/learning/leave_study_out_cv_condor_primary.py {project_dir}/learning/leave_study_out_cv_config/{env_name}/{cv_config}.json {env_dir} train_set.untampered_bulk_primary_cells_with_data -o {{output}} -r {tmp_dir}//leave_study_out_cv_condor/train_set.untampered_bulk_primary_cells_with_data/{cv_config}'.format(
                project_dir=PROJECT_DIR,
                env_name=ENV_NAME,
                env_dir=ENV_DIR,
                tmp_dir=TMP_DIR,
                cv_config=BNC_LOG_CPM_LINEAR_SVM_CONSTANT_PRIOR
            )
        ]
        for c in commands:
            shell(c)


rule bnc_logistic_full_constant_leave_study_out_cv_condor:
    input:
        data_set='{}/data/data_set_metadata.json'.format(ENV_DIR),
        exp_list_f='{}/data/experiment_lists/train_set.untampered_bulk_primary_cells_with_data/experiment_list.json'.format(ENV_DIR),
        labelling_f='{}/data/experiment_lists/train_set.untampered_bulk_primary_cells_with_data/labelling.json'.format(ENV_DIR)
    output:
        '{env_dir}/data/experiment_lists/train_set.untampered_bulk_primary_cells_with_data/leave_study_out_cv_results/{cv_config}/leave_study_out_cv_results.json'.format(
            env_dir=ENV_DIR,
            cv_config=BNC_LOG_CPM_LOGISTIC_L2_CONSTANT_PRIOR
        )
    run:
        commands = [
            'mkdir -p {tmp_dir}/leave_study_out_cv_condor/train_set.untampered_bulk_primary_cells_with_data/{cv_config}'.format(
                tmp_dir=TMP_DIR,
                cv_config=BNC_LOG_CPM_LOGISTIC_L2_CONSTANT_PRIOR
            ),
            'mkdir -p {env_dir}/data/experiment_lists/train_set.untampered_bulk_primary_cells_with_data/leave_study_out_cv_results/{cv_config}'.format(
                env_dir=ENV_DIR,
                cv_config=BNC_LOG_CPM_LOGISTIC_L2_CONSTANT_PRIOR
            ),
            'python2.7 {project_dir}/learning/leave_study_out_cv_condor_primary.py {project_dir}/learning/leave_study_out_cv_config/{env_name}/{cv_config}.json {env_dir} train_set.untampered_bulk_primary_cells_with_data -o {{output}} -r {tmp_dir}//leave_study_out_cv_condor/train_set.untampered_bulk_primary_cells_with_data/{cv_config}'.format(
                project_dir=PROJECT_DIR,
                env_name=ENV_NAME,
                env_dir=ENV_DIR,
                tmp_dir=TMP_DIR,
                cv_config=BNC_LOG_CPM_LOGISTIC_L2_CONSTANT_PRIOR
            )
        ]
        for c in commands:
            shell(c)


rule bnc_logistic_full_leave_study_out_cv_condor:
    input:
        data_set='{}/data/data_set_metadata.json'.format(ENV_DIR),
        exp_list_f='{}/data/experiment_lists/train_set.untampered_bulk_primary_cells_with_data/experiment_list.json'.format(ENV_DIR),
        labelling_f='{}/data/experiment_lists/train_set.untampered_bulk_primary_cells_with_data/labelling.json'.format(ENV_DIR)
    output:
        '{env_dir}/data/experiment_lists/train_set.untampered_bulk_primary_cells_with_data/leave_study_out_cv_results/{cv_config}/leave_study_out_cv_results.json'.format(
            env_dir=ENV_DIR,
            cv_config=BNC_LOG_CPM_LOGISTIC_L2
        )
    run:
        commands = [
            'mkdir -p {tmp_dir}/leave_study_out_cv_condor/train_set.untampered_bulk_primary_cells_with_data/{cv_config}'.format(
                tmp_dir=TMP_DIR,
                cv_config=BNC_LOG_CPM_LOGISTIC_L2
            ),
            'mkdir -p {env_dir}/data/experiment_lists/train_set.untampered_bulk_primary_cells_with_data/leave_study_out_cv_results/{cv_config}'.format(
                env_dir=ENV_DIR,
                cv_config=BNC_LOG_CPM_LOGISTIC_L2
            ),
            'python2.7 {project_dir}/learning/leave_study_out_cv_condor_primary.py {project_dir}/learning/leave_study_out_cv_config/{env_name}/{cv_config}.json {env_dir} train_set.untampered_bulk_primary_cells_with_data -o {{output}} -r {tmp_dir}//leave_study_out_cv_condor/train_set.untampered_bulk_primary_cells_with_data/{cv_config}'.format(
                project_dir=PROJECT_DIR,
                env_name=ENV_NAME,
                env_dir=ENV_DIR,
                tmp_dir=TMP_DIR,
                cv_config=BNC_LOG_CPM_LOGISTIC_L2
            )
        ]
        for c in commands:
            shell(c)

rule bnc_logistic_psudeo_0_1_full_leave_study_out_cv_condor:
    input:
        data_set='{}/data/data_set_metadata.json'.format(ENV_DIR),
        exp_list_f='{}/data/experiment_lists/train_set.untampered_bulk_primary_cells_with_data/experiment_list.json'.format(ENV_DIR),
        labelling_f='{}/data/experiment_lists/train_set.untampered_bulk_primary_cells_with_data/labelling.json'.format(ENV_DIR)
    output:
        '{env_dir}/data/experiment_lists/train_set.untampered_bulk_primary_cells_with_data/leave_study_out_cv_results/{cv_config}/leave_study_out_cv_results.json'.format(
            env_dir=ENV_DIR,
            cv_config=BNC_LOG_CPM_LOGISTIC_L2_PSEUDO_0_1
        )
    run:
        commands = [
            'mkdir -p {tmp_dir}/leave_study_out_cv_condor/train_set.untampered_bulk_primary_cells_with_data/{cv_config}'.format(
                tmp_dir=TMP_DIR,
                cv_config=BNC_LOG_CPM_LOGISTIC_L2_PSEUDO_0_1
            ),
            'mkdir -p {env_dir}/data/experiment_lists/train_set.untampered_bulk_primary_cells_with_data/leave_study_out_cv_results/{cv_config}'.format(
                env_dir=ENV_DIR,
                cv_config=BNC_LOG_CPM_LOGISTIC_L2_PSEUDO_0_1
            ),
            'python2.7 {project_dir}/learning/leave_study_out_cv_condor_primary.py {project_dir}/learning/leave_study_out_cv_config/{env_name}/{cv_config}.json {env_dir} train_set.untampered_bulk_primary_cells_with_data -o {{output}} -r {tmp_dir}//leave_study_out_cv_condor/train_set.untampered_bulk_primary_cells_with_data/{cv_config}'.format(
                project_dir=PROJECT_DIR,
                env_name=ENV_NAME,
                env_dir=ENV_DIR,
                tmp_dir=TMP_DIR,
                cv_config=BNC_LOG_CPM_LOGISTIC_L2_PSEUDO_0_1
            )
        ]
        for c in commands:
            shell(c)

rule bnc_logistic_conditional_prior_full_leave_study_out_cv_condor:
    input:
        data_set='{}/data/data_set_metadata.json'.format(ENV_DIR),
        exp_list_f='{}/data/experiment_lists/train_set.untampered_bulk_primary_cells_with_data/experiment_list.json'.format(ENV_DIR),
        labelling_f='{}/data/experiment_lists/train_set.untampered_bulk_primary_cells_with_data/labelling.json'.format(ENV_DIR)
    output:
        '{env_dir}/data/experiment_lists/train_set.untampered_bulk_primary_cells_with_data/leave_study_out_cv_results/{cv_config}/leave_study_out_cv_results.json'.format(
            env_dir=ENV_DIR,
            cv_config=BNC_LOG_CPM_LOGISTIC_L2_COND
        )
    run:
        commands = [
            'mkdir -p {tmp_dir}/leave_study_out_cv_condor/train_set.untampered_bulk_primary_cells_with_data/{cv_config}'.format(
                tmp_dir=TMP_DIR,
                cv_config=BNC_LOG_CPM_LOGISTIC_L2_COND
            ),
            'mkdir -p {env_dir}/data/experiment_lists/train_set.untampered_bulk_primary_cells_with_data/leave_study_out_cv_results/{cv_config}'.format(
                env_dir=ENV_DIR,
                cv_config=BNC_LOG_CPM_LOGISTIC_L2_COND
            ),
            'python2.7 {project_dir}/learning/leave_study_out_cv_condor_primary.py {project_dir}/learning/leave_study_out_cv_config/{env_name}/{cv_config}.json {env_dir} train_set.untampered_bulk_primary_cells_with_data -o {{output}} -r {tmp_dir}//leave_study_out_cv_condor/train_set.untampered_bulk_primary_cells_with_data/{cv_config}'.format(
                project_dir=PROJECT_DIR,
                env_name=ENV_NAME,
                env_dir=ENV_DIR,
                tmp_dir=TMP_DIR,
                cv_config=BNC_LOG_CPM_LOGISTIC_L2_COND
            )
        ]
        for c in commands:
            shell(c)



rule bnc_logistic_conditional_prior_static_bins_full_leave_study_out_cv_condor:
    input:
        data_set='{}/data/data_set_metadata.json'.format(ENV_DIR),
        exp_list_f='{}/data/experiment_lists/train_set.untampered_bulk_primary_cells_with_data/experiment_list.json'.format(ENV_DIR),
        labelling_f='{}/data/experiment_lists/train_set.untampered_bulk_primary_cells_with_data/labelling.json'.format(ENV_DIR)
    output: 
        '{env_dir}/data/experiment_lists/train_set.untampered_bulk_primary_cells_with_data/leave_study_out_cv_results/{cv_config}/leave_study_out_cv_results.json'.format(
            env_dir=ENV_DIR,
            cv_config=BNC_LOG_CPM_LOGISTIC_L2_COND_STATIC_BINS
        )   
    run:    
        commands = [
            'mkdir -p {tmp_dir}/leave_study_out_cv_condor/train_set.untampered_bulk_primary_cells_with_data/{cv_config}'.format(
                tmp_dir=TMP_DIR,
                cv_config=BNC_LOG_CPM_LOGISTIC_L2_COND_STATIC_BINS
            ),
            'mkdir -p {env_dir}/data/experiment_lists/train_set.untampered_bulk_primary_cells_with_data/leave_study_out_cv_results/{cv_config}'.format(
                env_dir=ENV_DIR,
                cv_config=BNC_LOG_CPM_LOGISTIC_L2_COND_STATIC_BINS
            ),
            'python2.7 {project_dir}/learning/leave_study_out_cv_condor_primary.py {project_dir}/learning/leave_study_out_cv_config/{env_name}/{cv_config}.json {env_dir} train_set.untampered_bulk_primary_cells_with_data -o {{output}} -r {tmp_dir}/leave_study_out_cv_condor/train_set.untampered_bulk_primary_cells_with_data/{cv_config}'.format(
                project_dir=PROJECT_DIR,
                env_name=ENV_NAME,
                env_dir=ENV_DIR,
                tmp_dir=TMP_DIR,
                cv_config=BNC_LOG_CPM_LOGISTIC_L2_COND_STATIC_BINS
            )
        ]
        for c in commands:
            shell(c)

rule bnc_linear_svm_conditional_prior_full_leave_study_out_cv_condor:
    input:
        data_set='{}/data/data_set_metadata.json'.format(ENV_DIR),
        exp_list_f='{}/data/experiment_lists/train_set.untampered_bulk_primary_cells_with_data/experiment_list.json'.format(ENV_DIR),
        labelling_f='{}/data/experiment_lists/train_set.untampered_bulk_primary_cells_with_data/labelling.json'.format(ENV_DIR)
    output:
        '{env_dir}/data/experiment_lists/train_set.untampered_bulk_primary_cells_with_data/leave_study_out_cv_results/{cv_config}/leave_study_out_cv_results.json'.format(
            env_dir=ENV_DIR,
            cv_config=BNC_LOG_CPM_LINEAR_SVM_COND
        )
    run:
        commands = [
            'mkdir -p {tmp_dir}/leave_study_out_cv_condor/train_set.untampered_bulk_primary_cells_with_data/{cv_config}'.format(
                tmp_dir=TMP_DIR,
                cv_config=BNC_LOG_CPM_LINEAR_SVM_COND
            ),
            'mkdir -p {env_dir}/data/experiment_lists/train_set.untampered_bulk_primary_cells_with_data/leave_study_out_cv_results/{cv_config}'.format(
                env_dir=ENV_DIR,
                cv_config=BNC_LOG_CPM_LINEAR_SVM_COND
            ),
            'python2.7 {project_dir}/learning/leave_study_out_cv_condor_primary.py {project_dir}/learning/leave_study_out_cv_config/{env_name}/{cv_config}.json {env_dir} train_set.untampered_bulk_primary_cells_with_data -o {{output}} -r {tmp_dir}//leave_study_out_cv_condor/train_set.untampered_bulk_primary_cells_with_data/{cv_config}'.format(
                project_dir=PROJECT_DIR,
                env_name=ENV_NAME,
                env_dir=ENV_DIR,
                tmp_dir=TMP_DIR,
                cv_config=BNC_LOG_CPM_LINEAR_SVM_COND
            )
        ]
        for c in commands:
            shell(c)

rule isotonic_logistic_full_leave_study_out_cv_condor:
    input:
        data_set='{}/data/data_set_metadata.json'.format(ENV_DIR),
        exp_list_f='{}/data/experiment_lists/train_set.untampered_bulk_primary_cells_with_data/experiment_list.json'.format(ENV_DIR),
        labelling_f='{}/data/experiment_lists/train_set.untampered_bulk_primary_cells_with_data/labelling.json'.format(ENV_DIR)
    output:
        '{env_dir}/data/experiment_lists/train_set.untampered_bulk_primary_cells_with_data/leave_study_out_cv_results/{cv_config}/leave_study_out_cv_results.json'.format(
            env_dir=ENV_DIR,
            cv_config=ISOTONIC_LOGISTIC_L2_LOG_CPM_FULL
        )
    run:
        commands = [
            'mkdir -p {tmp_dir}/leave_study_out_cv_condor/train_set.untampered_bulk_primary_cells_with_data/{cv_config}'.format(
                tmp_dir=TMP_DIR,
                cv_config=ISOTONIC_LOGISTIC_L2_LOG_CPM_FULL
            ),
            'mkdir -p {env_dir}/data/experiment_lists/train_set.untampered_bulk_primary_cells_with_data/leave_study_out_cv_results/{cv_config}'.format(
                env_dir=ENV_DIR,
                cv_config=ISOTONIC_LOGISTIC_L2_LOG_CPM_FULL
            ),
            'python2.7 {project_dir}/learning/leave_study_out_cv_condor_primary.py {project_dir}/learning/leave_study_out_cv_config/{env_name}/{cv_config}.json {env_dir} train_set.untampered_bulk_primary_cells_with_data -o {{output}} -r {tmp_dir}//leave_study_out_cv_condor/train_set.untampered_bulk_primary_cells_with_data/{cv_config}'.format(
                project_dir=PROJECT_DIR,
                env_name=ENV_NAME,
                env_dir=ENV_DIR,
                tmp_dir=TMP_DIR,
                cv_config=ISOTONIC_LOGISTIC_L2_LOG_CPM_FULL
            )
        ]
        for c in commands:
            shell(c) 

rule isotonic_logistic_assert_ambig_neg_full_leave_study_out_cv_condor:
    input:
        data_set='{}/data/data_set_metadata.json'.format(ENV_DIR),
        exp_list_f='{}/data/experiment_lists/train_set.untampered_bulk_primary_cells_with_data/experiment_list.json'.format(ENV_DIR),
        labelling_f='{}/data/experiment_lists/train_set.untampered_bulk_primary_cells_with_data/labelling.json'.format(ENV_DIR)
    output:
        '{env_dir}/data/experiment_lists/train_set.untampered_bulk_primary_cells_with_data/leave_study_out_cv_results/{cv_config}/leave_study_out_cv_results.json'.format(
            env_dir=ENV_DIR,
            cv_config=ISOTONIC_LOGISTIC_L2_LOG_CPM_ASSERT_AMBIG_NEG_FULL
        )
    run:
        commands = [
            'mkdir -p {tmp_dir}/leave_study_out_cv_condor/train_set.untampered_bulk_primary_cells_with_data/{cv_config}'.format(
                tmp_dir=TMP_DIR,
                cv_config=ISOTONIC_LOGISTIC_L2_LOG_CPM_ASSERT_AMBIG_NEG_FULL
            ),
            'mkdir -p {env_dir}/data/experiment_lists/train_set.untampered_bulk_primary_cells_with_data/leave_study_out_cv_results/{cv_config}'.format(
                env_dir=ENV_DIR,
                cv_config=ISOTONIC_LOGISTIC_L2_LOG_CPM_ASSERT_AMBIG_NEG_FULL
            ),
            'python2.7 {project_dir}/learning/leave_study_out_cv_condor_primary.py {project_dir}/learning/leave_study_out_cv_config/{env_name}/{cv_config}.json {env_dir} train_set.untampered_bulk_primary_cells_with_data -o {{output}} -r {tmp_dir}//leave_study_out_cv_condor/train_set.untampered_bulk_primary_cells_with_data/{cv_config}'.format(
                project_dir=PROJECT_DIR,
                env_name=ENV_NAME,
                env_dir=ENV_DIR,
                tmp_dir=TMP_DIR,
                cv_config=ISOTONIC_LOGISTIC_L2_LOG_CPM_ASSERT_AMBIG_NEG_FULL
            )
        ]
        for c in commands:
            shell(c)

rule tpr_logistic_assert_ambig_neg_full_leave_study_out_cv_condor:
    input:
        data_set='{}/data/data_set_metadata.json'.format(ENV_DIR),
        exp_list_f='{}/data/experiment_lists/train_set.untampered_bulk_primary_cells_with_data/experiment_list.json'.format(ENV_DIR),
        labelling_f='{}/data/experiment_lists/train_set.untampered_bulk_primary_cells_with_data/labelling.json'.format(ENV_DIR)
    output:
        '{env_dir}/data/experiment_lists/train_set.untampered_bulk_primary_cells_with_data/leave_study_out_cv_results/{cv_config}/leave_study_out_cv_results.json'.format(
            env_dir=ENV_DIR,
            cv_config=TPR_LOGISTIC_L2_LOG_CPM_FULL
        )
    run:
        commands = [
            'mkdir -p {tmp_dir}/leave_study_out_cv_condor/train_set.untampered_bulk_primary_cells_with_data/{cv_config}'.format(
                tmp_dir=TMP_DIR,
                cv_config=TPR_LOGISTIC_L2_LOG_CPM_FULL
            ),
            'mkdir -p {env_dir}/data/experiment_lists/train_set.untampered_bulk_primary_cells_with_data/leave_study_out_cv_results/{cv_config}'.format(
                env_dir=ENV_DIR,
                cv_config=TPR_LOGISTIC_L2_LOG_CPM_FULL
            ),
            'python2.7 {project_dir}/learning/leave_study_out_cv_condor_primary.py {project_dir}/learning/leave_study_out_cv_config/{env_name}/{cv_config}.json {env_dir} train_set.untampered_bulk_primary_cells_with_data -o {{output}} -r {tmp_dir}//leave_study_out_cv_condor/train_set.untampered_bulk_primary_cells_with_data/{cv_config}'.format(
                project_dir=PROJECT_DIR,
                env_name=ENV_NAME,
                env_dir=ENV_DIR,
                tmp_dir=TMP_DIR,
                cv_config=TPR_LOGISTIC_L2_LOG_CPM_FULL
            )
        ]
        for c in commands:
            shell(c)

rule isotonic_logistic_full_downweight_leave_study_out_cv_condor:
    input:
        data_set='{}/data/data_set_metadata.json'.format(ENV_DIR),
        exp_list_f='{}/data/experiment_lists/train_set.untampered_bulk_primary_cells_with_data/experiment_list.json'.format(ENV_DIR),
        labelling_f='{}/data/experiment_lists/train_set.untampered_bulk_primary_cells_with_data/labelling.json'.format(ENV_DIR)
    output:
        '{env_dir}/data/experiment_lists/train_set.untampered_bulk_primary_cells_with_data/leave_study_out_cv_results/{cv_config}/leave_study_out_cv_results.json'.format(
            env_dir=ENV_DIR,
            cv_config=ISOTONIC_LOGISTIC_L2_LOG_CPM_FULL_DOWNWEIGHT
        )
    run:
        commands = [
            'mkdir -p {tmp_dir}/leave_study_out_cv_condor/train_set.untampered_bulk_primary_cells_with_data/{cv_config}'.format(
                tmp_dir=TMP_DIR,
                cv_config=ISOTONIC_LOGISTIC_L2_LOG_CPM_FULL_DOWNWEIGHT
            ),
            'mkdir -p {env_dir}/data/experiment_lists/train_set.untampered_bulk_primary_cells_with_data/leave_study_out_cv_results/{cv_config}'.format(
                env_dir=ENV_DIR,
                cv_config=ISOTONIC_LOGISTIC_L2_LOG_CPM_FULL_DOWNWEIGHT
            ),
            'python2.7 {project_dir}/learning/leave_study_out_cv_condor_primary.py {project_dir}/learning/leave_study_out_cv_config/{env_name}/{cv_config}.json {env_dir} train_set.untampered_bulk_primary_cells_with_data -o {{output}} -r {tmp_dir}//leave_study_out_cv_condor/train_set.untampered_bulk_primary_cells_with_data/{cv_config}'.format(
                project_dir=PROJECT_DIR,
                env_name=ENV_NAME,
                env_dir=ENV_DIR,
                tmp_dir=TMP_DIR,
                cv_config=ISOTONIC_LOGISTIC_L2_LOG_CPM_FULL_DOWNWEIGHT
            )
        ]
        for c in commands:
            shell(c)


rule isotonic_logistic_full_assert_ambig_neg_downweight_leave_study_out_cv_condor:
    input:
        data_set='{}/data/data_set_metadata.json'.format(ENV_DIR),
        exp_list_f='{}/data/experiment_lists/train_set.untampered_bulk_primary_cells_with_data/experiment_list.json'.format(ENV_DIR),
        labelling_f='{}/data/experiment_lists/train_set.untampered_bulk_primary_cells_with_data/labelling.json'.format(ENV_DIR)
    output:
        '{env_dir}/data/experiment_lists/train_set.untampered_bulk_primary_cells_with_data/leave_study_out_cv_results/{cv_config}/leave_study_out_cv_results.json'.format(
            env_dir=ENV_DIR,
            cv_config=ISOTONIC_LOGISTIC_L2_LOG_CPM_ASSERT_AMBIG_NEG_FULL_DOWNWEIGHT
        )
    run:
        commands = [
            'mkdir -p {tmp_dir}/leave_study_out_cv_condor/train_set.untampered_bulk_primary_cells_with_data/{cv_config}'.format(
                tmp_dir=TMP_DIR,
                cv_config=ISOTONIC_LOGISTIC_L2_LOG_CPM_ASSERT_AMBIG_NEG_FULL_DOWNWEIGHT
            ),
            'mkdir -p {env_dir}/data/experiment_lists/train_set.untampered_bulk_primary_cells_with_data/leave_study_out_cv_results/{cv_config}'.format(
                env_dir=ENV_DIR,
                cv_config=ISOTONIC_LOGISTIC_L2_LOG_CPM_ASSERT_AMBIG_NEG_FULL_DOWNWEIGHT
            ),
            'python2.7 {project_dir}/learning/leave_study_out_cv_condor_primary.py {project_dir}/learning/leave_study_out_cv_config/{env_name}/{cv_config}.json {env_dir} train_set.untampered_bulk_primary_cells_with_data -o {{output}} -r {tmp_dir}//leave_study_out_cv_condor/train_set.untampered_bulk_primary_cells_with_data/{cv_config}'.format(
                project_dir=PROJECT_DIR,
                env_name=ENV_NAME,
                env_dir=ENV_DIR,
                tmp_dir=TMP_DIR,
                cv_config=ISOTONIC_LOGISTIC_L2_LOG_CPM_ASSERT_AMBIG_NEG_FULL_DOWNWEIGHT
            )
        ]
        for c in commands:
            shell(c)

rule cdc_assert_ambig_neg_collapse_leave_study_out_cv:
    input:
        data_set='{}/data/data_set_metadata.json'.format(ENV_DIR),
        exp_list_f='{}/data/experiment_lists/train_set.untampered_bulk_primary_cells_with_data/experiment_list.json'.format(ENV_DIR),
        labelling_f='{}/data/experiment_lists/train_set.untampered_bulk_primary_cells_with_data/labelling.json'.format(ENV_DIR)
    output:
        '{env_dir}/data/experiment_lists/train_set.untampered_bulk_primary_cells_with_data/leave_study_out_cv_results/{cv_config}/leave_study_out_cv_results.json'.format(
            env_dir=ENV_DIR,
            cv_config=CDC_LOG_CPM_LOGISTIC_L2_ASSERT_AMBIG_NEG_COLLAPSE
        )
    run:
        commands = [
            'mkdir -p {tmp_dir}/leave_study_out_cv_batches/train_set.untampered_bulk_primary_cells_with_data/{cv_config}'.format(
                tmp_dir=TMP_DIR,
                cv_config=CDC_LOG_CPM_LOGISTIC_L2_ASSERT_AMBIG_NEG_COLLAPSE
            ),
            'mkdir -p {env_dir}/data/experiment_lists/train_set.untampered_bulk_primary_cells_with_data/leave_study_out_cv_results/{cv_config}'.format(
                env_dir=ENV_DIR,
                cv_config=CDC_LOG_CPM_LOGISTIC_L2_ASSERT_AMBIG_NEG_COLLAPSE
            ),
            'python2.7 {project_dir}/learning/leave_study_out_cv.py {project_dir}/learning/leave_study_out_cv_config/{env_name}/{cv_config}.json {env_dir} train_set.untampered_bulk_primary_cells_with_data -b {tmp_dir}/leave_study_out_cv_batches/train_set.untampered_bulk_primary_cells_with_data/{cv_config} -o {{output}}'.format(
                project_dir=PROJECT_DIR,
                env_name=ENV_NAME,
                env_dir=ENV_DIR,
                tmp_dir=TMP_DIR,
                cv_config=CDC_LOG_CPM_LOGISTIC_L2_ASSERT_AMBIG_NEG_COLLAPSE
            )
        ]
        for c in commands:
            shell(c)

rule cdc_assert_ambig_neg_collapse_downweight_leave_study_out_cv:
    input:
        data_set='{}/data/data_set_metadata.json'.format(ENV_DIR),
        exp_list_f='{}/data/experiment_lists/train_set.untampered_bulk_primary_cells_with_data/experiment_list.json'.format(ENV_DIR),
        labelling_f='{}/data/experiment_lists/train_set.untampered_bulk_primary_cells_with_data/labelling.json'.format(ENV_DIR)
    output:
        '{env_dir}/data/experiment_lists/train_set.untampered_bulk_primary_cells_with_data/leave_study_out_cv_results/{cv_config}/leave_study_out_cv_results.json'.format(
            env_dir=ENV_DIR,
            cv_config=CDC_LOG_CPM_LOGISTIC_L2_ASSERT_AMBIG_NEG_COLLAPSE_DOWNWEIGHT
        )
    run:
        commands = [
            'mkdir -p {tmp_dir}/leave_study_out_cv_batches/train_set.untampered_bulk_primary_cells_with_data/{cv_config}'.format(
                tmp_dir=TMP_DIR,
                cv_config=CDC_LOG_CPM_LOGISTIC_L2_ASSERT_AMBIG_NEG_COLLAPSE_DOWNWEIGHT
            ),
            'mkdir -p {env_dir}/data/experiment_lists/train_set.untampered_bulk_primary_cells_with_data/leave_study_out_cv_results/{cv_config}'.format(
                env_dir=ENV_DIR,
                cv_config=CDC_LOG_CPM_LOGISTIC_L2_ASSERT_AMBIG_NEG_COLLAPSE_DOWNWEIGHT
            ),
            'python2.7 {project_dir}/learning/leave_study_out_cv.py {project_dir}/learning/leave_study_out_cv_config/{env_name}/{cv_config}.json {env_dir} train_set.untampered_bulk_primary_cells_with_data -b {tmp_dir}/leave_study_out_cv_batches/train_set.untampered_bulk_primary_cells_with_data/{cv_config} -o {{output}}'.format(
                project_dir=PROJECT_DIR,
                env_name=ENV_NAME,
                env_dir=ENV_DIR,
                tmp_dir=TMP_DIR,
                cv_config=CDC_LOG_CPM_LOGISTIC_L2_ASSERT_AMBIG_NEG_COLLAPSE_DOWNWEIGHT
            )
        ]
        for c in commands:
            shell(c)


rule cdc_assert_ambig_neg_collapse_downweight_leave_study_out_cv_condorized:
    input:
        data_set='{}/data/data_set_metadata.json'.format(ENV_DIR),
        exp_list_f='{}/data/experiment_lists/train_set.untampered_bulk_primary_cells_with_data/experiment_list.json'.format(ENV_DIR),
        labelling_f='{}/data/experiment_lists/train_set.untampered_bulk_primary_cells_with_data/labelling.json'.format(ENV_DIR)
    output:
        '{env_dir}/data/experiment_lists/train_set.untampered_bulk_primary_cells_with_data/leave_study_out_cv_results/{cv_config}/leave_study_out_cv_results.json'.format(
            env_dir=ENV_DIR,
            cv_config=CDC_LOG_CPM_LOGISTIC_L2_ASSERT_AMBIG_NEG_COLLAPSE_DOWNWEIGHT
        )
    run:
        commands = [
            'mkdir -p {tmp_dir}/leave_study_out_cv_condor/train_set.untampered_bulk_primary_cells_with_data/{cv_config}'.format(
                tmp_dir=TMP_DIR,
                cv_config=CDC_LOG_CPM_LOGISTIC_L2_ASSERT_AMBIG_NEG_COLLAPSE_DOWNWEIGHT
            ),
            'mkdir -p {env_dir}/data/experiment_lists/train_set.untampered_bulk_primary_cells_with_data/leave_study_out_cv_results/{cv_config}'.format(
                env_dir=ENV_DIR,
                cv_config=CDC_LOG_CPM_LOGISTIC_L2_ASSERT_AMBIG_NEG_COLLAPSE_DOWNWEIGHT
            ),
            'python2.7 {project_dir}/learning/leave_study_out_cv_condor_primary.py {project_dir}/learning/leave_study_out_cv_config/{env_name}/{cv_config}.json {env_dir} train_set.untampered_bulk_primary_cells_with_data -o {{output}} -r {tmp_dir}//leave_study_out_cv_condor/train_set.untampered_bulk_primary_cells_with_data/{cv_config}'.format(
                project_dir=PROJECT_DIR,
                env_name=ENV_NAME,
                env_dir=ENV_DIR,
                tmp_dir=TMP_DIR,
                cv_config=CDC_LOG_CPM_LOGISTIC_L2_ASSERT_AMBIG_NEG_COLLAPSE_DOWNWEIGHT
            )
        ]
        for c in commands:
            shell(c)

rule cdc_assert_ambig_neg_full_downweight_leave_study_out_cv_condorized:
    input:
        data_set='{}/data/data_set_metadata.json'.format(ENV_DIR),
        exp_list_f='{}/data/experiment_lists/train_set.untampered_bulk_primary_cells_with_data/experiment_list.json'.format(ENV_DIR),
        labelling_f='{}/data/experiment_lists/train_set.untampered_bulk_primary_cells_with_data/labelling.json'.format(ENV_DIR)
    output:
        '{env_dir}/data/experiment_lists/train_set.untampered_bulk_primary_cells_with_data/leave_study_out_cv_results/{cv_config}/leave_study_out_cv_results.json'.format(
            env_dir=ENV_DIR,
            cv_config=CDC_LOG_CPM_LOGISTIC_L2_ASSERT_AMBIG_NEG_FULL_DOWNWEIGHT
        )
    run:
        commands = [
            'mkdir -p {tmp_dir}/leave_study_out_cv_condor/train_set.untampered_bulk_primary_cells_with_data/{cv_config}'.format(
                tmp_dir=TMP_DIR,
                cv_config=CDC_LOG_CPM_LOGISTIC_L2_ASSERT_AMBIG_NEG_FULL_DOWNWEIGHT
            ),
            'mkdir -p {env_dir}/data/experiment_lists/train_set.untampered_bulk_primary_cells_with_data/leave_study_out_cv_results/{cv_config}'.format(
                env_dir=ENV_DIR,
                cv_config=CDC_LOG_CPM_LOGISTIC_L2_ASSERT_AMBIG_NEG_FULL_DOWNWEIGHT
            ),
            'python2.7 {project_dir}/learning/leave_study_out_cv_condor_primary.py {project_dir}/learning/leave_study_out_cv_config/{env_name}/{cv_config}.json {env_dir} train_set.untampered_bulk_primary_cells_with_data -o {{output}} -r {tmp_dir}//leave_study_out_cv_condor/train_set.untampered_bulk_primary_cells_with_data/{cv_config}'.format(
                project_dir=PROJECT_DIR,
                env_name=ENV_NAME,
                env_dir=ENV_DIR,
                tmp_dir=TMP_DIR,
                cv_config=CDC_LOG_CPM_LOGISTIC_L2_ASSERT_AMBIG_NEG_FULL_DOWNWEIGHT
            )
        ]
        for c in commands:
            shell(c)

rule cdc_assert_ambig_neg_full_leave_study_out_cv_condor:
    input:
        data_set='{}/data/data_set_metadata.json'.format(ENV_DIR),
        exp_list_f='{}/data/experiment_lists/train_set.untampered_bulk_primary_cells_with_data/experiment_list.json'.format(ENV_DIR),
        labelling_f='{}/data/experiment_lists/train_set.untampered_bulk_primary_cells_with_data/labelling.json'.format(ENV_DIR)
    output:
        '{env_dir}/data/experiment_lists/train_set.untampered_bulk_primary_cells_with_data/leave_study_out_cv_results/{cv_config}/leave_study_out_cv_results.json'.format(
            env_dir=ENV_DIR,
            cv_config=CDC_LOG_CPM_LOGISTIC_L2_ASSERT_AMBIG_NEG_FULL
        )
    run:
        commands = [
            'mkdir -p {tmp_dir}/leave_study_out_cv_condor/train_set.untampered_bulk_primary_cells_with_data/{cv_config}'.format(
                tmp_dir=TMP_DIR,
                cv_config=CDC_LOG_CPM_LOGISTIC_L2_ASSERT_AMBIG_NEG_FULL
            ),
            'mkdir -p {env_dir}/data/experiment_lists/train_set.untampered_bulk_primary_cells_with_data/leave_study_out_cv_results/{cv_config}'.format(
                env_dir=ENV_DIR,
                cv_config=CDC_LOG_CPM_LOGISTIC_L2_ASSERT_AMBIG_NEG_FULL
            ),
            'python2.7 {project_dir}/learning/leave_study_out_cv_condor_primary.py {project_dir}/learning/leave_study_out_cv_config/{env_name}/{cv_config}.json {env_dir} train_set.untampered_bulk_primary_cells_with_data -o {{output}} -r {tmp_dir}//leave_study_out_cv_condor/train_set.untampered_bulk_primary_cells_with_data/{cv_config}'.format(
                project_dir=PROJECT_DIR,
                env_name=ENV_NAME,
                env_dir=ENV_DIR,
                tmp_dir=TMP_DIR,
                cv_config=CDC_LOG_CPM_LOGISTIC_L2_ASSERT_AMBIG_NEG_FULL
            )
        ]
        for c in commands:
            shell(c)


rule cdc_remove_ambig_collapse_leave_study_out_cv:
    input:
        data_set='{}/data/data_set_metadata.json'.format(ENV_DIR),
        exp_list_f='{}/data/experiment_lists/train_set.untampered_bulk_primary_cells_with_data/experiment_list.json'.format(ENV_DIR),
        labelling_f='{}/data/experiment_lists/train_set.untampered_bulk_primary_cells_with_data/labelling.json'.format(ENV_DIR)
    output:
        '{env_dir}/data/experiment_lists/train_set.untampered_bulk_primary_cells_with_data/leave_study_out_cv_results/{cv_config}/leave_study_out_cv_results.json'.format(
            env_dir=ENV_DIR,
            cv_config=CDC_LOG_CPM_LOGISTIC_L2_REMOVE_AMBIG_COLLAPSE
        )
    run:
        commands = [
            'mkdir -p {tmp_dir}/leave_study_out_cv_batches/train_set.untampered_bulk_primary_cells_with_data/{cv_config}'.format(
                tmp_dir=TMP_DIR,
                cv_config=CDC_LOG_CPM_LOGISTIC_L2_REMOVE_AMBIG_COLLAPSE
            ),
            'mkdir -p {env_dir}/data/experiment_lists/train_set.untampered_bulk_primary_cells_with_data/leave_study_out_cv_results/{cv_config}'.format(
                env_dir=ENV_DIR,
                cv_config=CDC_LOG_CPM_LOGISTIC_L2_REMOVE_AMBIG_COLLAPSE
            ),
            'python2.7 {project_dir}/learning/leave_study_out_cv.py {project_dir}/learning/leave_study_out_cv_config/{env_name}/{cv_config}.json {env_dir} train_set.untampered_bulk_primary_cells_with_data -b {tmp_dir}/leave_study_out_cv_batches/train_set.untampered_bulk_primary_cells_with_data/{cv_config} -o {{output}}'.format(
                project_dir=PROJECT_DIR,
                env_name=ENV_NAME,
                env_dir=ENV_DIR,
                tmp_dir=TMP_DIR,
                cv_config=CDC_LOG_CPM_LOGISTIC_L2_REMOVE_AMBIG_COLLAPSE
            )
        ]
        for c in commands:
            shell(c)

rule per_label_full_leave_study_out_cv_condor:
    input:
        data_set='{}/data/data_set_metadata.json'.format(ENV_DIR),
        exp_list_f='{}/data/experiment_lists/train_set.untampered_bulk_primary_cells_with_data/experiment_list.json'.format(ENV_DIR),
        labelling_f='{}/data/experiment_lists/train_set.untampered_bulk_primary_cells_with_data/labelling.json'.format(ENV_DIR)
    output:
        '{env_dir}/data/experiment_lists/train_set.untampered_bulk_primary_cells_with_data/leave_study_out_cv_results/{cv_config}/leave_study_out_cv_results.json'.format(
            env_dir=ENV_DIR,
            cv_config=PER_LABEL_LOG_CPM_LOGISTIC_L2_FULL
        )
    run:
        commands = [
            'mkdir -p {tmp_dir}/leave_study_out_cv_condor/train_set.untampered_bulk_primary_cells_with_data/{cv_config}'.format(
                tmp_dir=TMP_DIR,
                cv_config=PER_LABEL_LOG_CPM_LOGISTIC_L2_FULL
            ),
            'mkdir -p {env_dir}/data/experiment_lists/train_set.untampered_bulk_primary_cells_with_data/leave_study_out_cv_results/{cv_config}'.format(
                env_dir=ENV_DIR,
                cv_config=PER_LABEL_LOG_CPM_LOGISTIC_L2_FULL
            ),
            'python2.7 {project_dir}/learning/leave_study_out_cv_condor_primary.py {project_dir}/learning/leave_study_out_cv_config/{env_name}/{cv_config}.json {env_dir} train_set.untampered_bulk_primary_cells_with_data -o {{output}} -r {tmp_dir}//leave_study_out_cv_condor/train_set.untampered_bulk_primary_cells_with_data/{cv_config}'.format(
                project_dir=PROJECT_DIR,
                env_name=ENV_NAME,
                env_dir=ENV_DIR,
                tmp_dir=TMP_DIR,
                cv_config=PER_LABEL_LOG_CPM_LOGISTIC_L2_FULL
            )
        ]
        for c in commands:
            shell(c)



rule one_nn_leave_study_out_cv_condor:
    input:
        data_set='{}/data/data_set_metadata.json'.format(ENV_DIR),
        exp_list_f='{}/data/experiment_lists/train_set.untampered_bulk_primary_cells_with_data/experiment_list.json'.format(ENV_DIR),
        labelling_f='{}/data/experiment_lists/train_set.untampered_bulk_primary_cells_with_data/labelling.json'.format(ENV_DIR)
    output:
        '{env_dir}/data/experiment_lists/train_set.untampered_bulk_primary_cells_with_data/leave_study_out_cv_results/{cv_config}/leave_study_out_cv_results.json'.format(
            env_dir=ENV_DIR,
            cv_config=ONE_NN_LOG_CPM_CORRELATION
        )
    run:
        commands = [    
            'mkdir -p {tmp_dir}/leave_study_out_cv_condor/train_set.untampered_bulk_primary_cells_with_data/{cv_config}'.format(
                tmp_dir=TMP_DIR,
                cv_config=ONE_NN_LOG_CPM_CORRELATION
            ),
            'mkdir -p {env_dir}/data/experiment_lists/train_set.untampered_bulk_primary_cells_with_data/leave_study_out_cv_results/{cv_config}'.format(
                env_dir=ENV_DIR,
                cv_config=ONE_NN_LOG_CPM_CORRELATION
            ),
            'python2.7 {project_dir}/learning/leave_study_out_cv_condor_primary.py {project_dir}/learning/leave_study_out_cv_config/{env_name}/{cv_config}.json {env_dir} train_set.untampered_bulk_primary_cells_with_data -o {{output}} -r {tmp_dir}//leave_study_out_cv_condor/train_set.untampered_bulk_primary_cells_with_data/{cv_config}'.format(
                project_dir=PROJECT_DIR,
                env_name=ENV_NAME,
                env_dir=ENV_DIR,
                tmp_dir=TMP_DIR,
                cv_config=ONE_NN_LOG_CPM_CORRELATION
            )
        ]
        for c in commands:
            shell(c)

#rule one_nn_leave_study_out_cv:
#    input:
#        data_set='{}/data/data_set_metadata.json'.format(ENV_DIR),
#        exp_list_f='{}/data/experiment_lists/train_set.untampered_bulk_primary_cells_with_data/experiment_list.json'.format(ENV_DIR),
#        labelling_f='{}/data/experiment_lists/train_set.untampered_bulk_primary_cells_with_data/labelling.json'.format(ENV_DIR)
#    output:
#        '{env_dir}/data/experiment_lists/train_set.untampered_bulk_primary_cells_with_data/leave_study_out_cv_results/{cv_config}/leave_study_out_cv_results.json'.format(
#            env_dir=ENV_DIR,
#            cv_config=ONE_NN_LOG_CPM_CORRELATION
#        )
#    run:
#        commands = [
#            'mkdir -p {tmp_dir}/leave_study_out_cv_batches/train_set.untampered_bulk_primary_cells_with_data/{cv_config}'.format(
#                tmp_dir=TMP_DIR,
#                cv_config=ONE_NN_LOG_CPM_CORRELATION
#            ),
#            'mkdir -p {env_dir}/data/experiment_lists/train_set.untampered_bulk_primary_cells_with_data/leave_study_out_cv_results/{cv_config}'.format(
#                env_dir=ENV_DIR,
#                cv_config=ONE_NN_LOG_CPM_CORRELATION
#            ),
#            'python2.7 {project_dir}/learning/leave_study_out_cv.py {project_dir}/learning/leave_study_out_cv_config/{env_name}/{cv_config}.json {env_dir} train_set.untampered_bulk_primary_cells_with_data -b {tmp_dir}/leave_study_out_cv_batches/train_set.untampered_bulk_primary_cells_with_data/{cv_config} -o {{output}}'.format(
#                project_dir=PROJECT_DIR,
#                env_name=ENV_NAME,
#                env_dir=ENV_DIR,
#                tmp_dir=TMP_DIR,
#                cv_config=ONE_NN_LOG_CPM_CORRELATION
#            )
#        ]
#        for c in commands:
#            shell(c)


#####################################################################
#   Test the various algorithms on the sets of experiments used
#   for testing the algorithms for bugs.
#####################################################################

rule toy_bnc_linear_svm_normal_conditional_prior_full_leave_study_out_cv_condor:
    input:
        data_set='{}/data/data_set_metadata.json'.format(ENV_DIR),
        exp_list_f='{}/data/experiment_lists/toy/experiment_list.json'.format(ENV_DIR),
        labelling_f='{}/data/experiment_lists/toy/labelling.json'.format(ENV_DIR)
    output:
        '{env_dir}/data/experiment_lists/toy/leave_study_out_cv_results/{cv_config}/leave_study_out_cv_results.json'.format(
            env_dir=ENV_DIR,
            cv_config=BNC_LOG_CPM_LINEAR_SVM_NORMAL
        )
    run:
        commands = [
            'mkdir -p {tmp_dir}/leave_study_out_cv_condor/toy/{cv_config}'.format(
                tmp_dir=TMP_DIR,
                cv_config=BNC_LOG_CPM_LINEAR_SVM_NORMAL
            ),
            'mkdir -p {env_dir}/data/experiment_lists/toy/leave_study_out_cv_results/{cv_config}'.format(
                env_dir=ENV_DIR,
                cv_config=BNC_LOG_CPM_LINEAR_SVM_NORMAL
            ),
            'python2.7 {project_dir}/learning/leave_study_out_cv_condor_primary.py {project_dir}/learning/leave_study_out_cv_config/{env_name}/{cv_config}.json {env_dir} toy -o {{output}} -r {tmp_dir}//leave_study_out_cv_condor/toy/{cv_config}'.format(
                project_dir=PROJECT_DIR,
                env_name=ENV_NAME,
                env_dir=ENV_DIR,
                tmp_dir=TMP_DIR,
                cv_config=BNC_LOG_CPM_LINEAR_SVM_NORMAL
            )
        ]
        for c in commands:
            shell(c)

rule toy_bnc_linear_svm_normal_fixed_std_conditional_prior_full_leave_study_out_cv_condor:
    input:
        data_set='{}/data/data_set_metadata.json'.format(ENV_DIR),
        exp_list_f='{}/data/experiment_lists/toy/experiment_list.json'.format(ENV_DIR),
        labelling_f='{}/data/experiment_lists/toy/labelling.json'.format(ENV_DIR)
    output:
        '{env_dir}/data/experiment_lists/toy/leave_study_out_cv_results/{cv_config}/leave_study_out_cv_results.json'.format(
            env_dir=ENV_DIR,
            cv_config=BNC_LOG_CPM_LINEAR_SVM_NORMAL_FIXED_STD
        )
    run:
        commands = [
            'mkdir -p {tmp_dir}/leave_study_out_cv_condor/toy/{cv_config}'.format(
                tmp_dir=TMP_DIR,
                cv_config=BNC_LOG_CPM_LINEAR_SVM_NORMAL_FIXED_STD
            ),
            'mkdir -p {env_dir}/data/experiment_lists/toy/leave_study_out_cv_results/{cv_config}'.format(
                env_dir=ENV_DIR,
                cv_config=BNC_LOG_CPM_LINEAR_SVM_NORMAL_FIXED_STD
            ),
            'python2.7 {project_dir}/learning/leave_study_out_cv_condor_primary.py {project_dir}/learning/leave_study_out_cv_config/{env_name}/{cv_config}.json {env_dir} toy -o {{output}} -r {tmp_dir}//leave_study_out_cv_condor/toy/{cv_config}'.format(
                project_dir=PROJECT_DIR,
                env_name=ENV_NAME,
                env_dir=ENV_DIR,
                tmp_dir=TMP_DIR,
                cv_config=BNC_LOG_CPM_LINEAR_SVM_NORMAL_FIXED_STD
            )
        ]
        for c in commands:
            shell(c)

rule toy_bnc_logistic_conditional_prior_full_leave_study_out_cv_condor:
    input:
        data_set='{}/data/data_set_metadata.json'.format(ENV_DIR),
        exp_list_f='{}/data/experiment_lists/toy/experiment_list.json'.format(ENV_DIR),
        labelling_f='{}/data/experiment_lists/toy/labelling.json'.format(ENV_DIR)
    output:
        '{env_dir}/data/experiment_lists/toy/leave_study_out_cv_results/{cv_config}/leave_study_out_cv_results.json'.format(
            env_dir=ENV_DIR,
            cv_config=BNC_LOG_CPM_LOGISTIC_L2_COND
        )
    run:
        commands = [
            'mkdir -p {tmp_dir}/leave_study_out_cv_condor/toy/{cv_config}'.format(
                tmp_dir=TMP_DIR,
                cv_config=BNC_LOG_CPM_LOGISTIC_L2_COND
            ),
            'mkdir -p {env_dir}/data/experiment_lists/toy/leave_study_out_cv_results/{cv_config}'.format(
                env_dir=ENV_DIR,
                cv_config=BNC_LOG_CPM_LOGISTIC_L2_COND
            ),
            'python2.7 {project_dir}/learning/leave_study_out_cv_condor_primary.py {project_dir}/learning/leave_study_out_cv_config/{env_name}/{cv_config}.json {env_dir} toy -o {{output}} -r {tmp_dir}//leave_study_out_cv_condor/toy/{cv_config}'.format(
                project_dir=PROJECT_DIR,
                env_name=ENV_NAME,
                env_dir=ENV_DIR,
                tmp_dir=TMP_DIR,
                cv_config=BNC_LOG_CPM_LOGISTIC_L2_COND
            )
        ]
        for c in commands:
            shell(c)

rule toy_bnc_logistic_conditional_prior_static_bins_full_leave_study_out_cv_condor:
    input:
        data_set='{}/data/data_set_metadata.json'.format(ENV_DIR),
        exp_list_f='{}/data/experiment_lists/toy/experiment_list.json'.format(ENV_DIR),
        labelling_f='{}/data/experiment_lists/toy/labelling.json'.format(ENV_DIR)
    output:
        '{env_dir}/data/experiment_lists/toy/leave_study_out_cv_results/{cv_config}/leave_study_out_cv_results.json'.format(
            env_dir=ENV_DIR,
            cv_config=BNC_LOG_CPM_LOGISTIC_L2_COND_STATIC_BINS
        )
    run:
        commands = [
            'mkdir -p {tmp_dir}/leave_study_out_cv_condor/toy/{cv_config}'.format(
                tmp_dir=TMP_DIR,
                cv_config=BNC_LOG_CPM_LOGISTIC_L2_COND_STATIC_BINS
            ),
            'mkdir -p {env_dir}/data/experiment_lists/toy/leave_study_out_cv_results/{cv_config}'.format(
                env_dir=ENV_DIR,
                cv_config=BNC_LOG_CPM_LOGISTIC_L2_COND_STATIC_BINS
            ),
            'python2.7 {project_dir}/learning/leave_study_out_cv_condor_primary.py {project_dir}/learning/leave_study_out_cv_config/{env_name}/{cv_config}.json {env_dir} toy -o {{output}} -r {tmp_dir}//leave_study_out_cv_condor/toy/{cv_config}'.format(
                project_dir=PROJECT_DIR,
                env_name=ENV_NAME,
                env_dir=ENV_DIR,
                tmp_dir=TMP_DIR,
                cv_config=BNC_LOG_CPM_LOGISTIC_L2_COND_STATIC_BINS
            )
        ]
        for c in commands:
            shell(c)

rule toy_cdc_remove_ambig_full_leave_study_out_cv: 
    input:
        data_set='{}/data/data_set_metadata.json'.format(ENV_DIR),
        exp_list_f='{}/data/experiment_lists/toy/experiment_list.json'.format(ENV_DIR),
        labelling_f='{}/data/experiment_lists/toy/labelling.json'.format(ENV_DIR)
    output:
        '{env_dir}/data/experiment_lists/toy/leave_study_out_cv_results/{cv_config}/leave_study_out_cv_results.json'.format(
            env_dir=ENV_DIR,
            cv_config=CDC_LOG_CPM_LOGISTIC_L2_REMOVE_AMBIG_FULL
        )
    run:
        commands = [
            'mkdir -p {tmp_dir}/leave_study_out_cv_condor/toy/{cv_config}'.format(
                tmp_dir=TMP_DIR,
                cv_config=CDC_LOG_CPM_LOGISTIC_L2_REMOVE_AMBIG_FULL
            ),
            'mkdir -p {env_dir}/data/experiment_lists/toy/leave_study_out_cv_results/{cv_config}'.format(
                env_dir=ENV_DIR,
                cv_config=CDC_LOG_CPM_LOGISTIC_L2_REMOVE_AMBIG_FULL
            ),
            'python2.7 {project_dir}/learning/leave_study_out_cv_condor_primary.py {project_dir}/learning/leave_study_out_cv_config/{env_name}/{cv_config}.json {env_dir} toy -o {{output}} -r {tmp_dir}//leave_study_out_cv_condor/toy/{cv_config}'.format(
                project_dir=PROJECT_DIR,
                env_name=ENV_NAME,
                env_dir=ENV_DIR,
                tmp_dir=TMP_DIR,
                cv_config=CDC_LOG_CPM_LOGISTIC_L2_REMOVE_AMBIG_FULL
            )
        ]
        for c in commands:
            shell(c)

rule toy_isotonic_logistic_full_leave_study_out_cv_condor:
    input:
        data_set='{}/data/data_set_metadata.json'.format(ENV_DIR),
        exp_list_f='{}/data/experiment_lists/toy/experiment_list.json'.format(ENV_DIR),
        labelling_f='{}/data/experiment_lists/toy/labelling.json'.format(ENV_DIR)
    output:
        '{env_dir}/data/experiment_lists/toy/leave_study_out_cv_results/{cv_config}/leave_study_out_cv_results.json'.format(
            env_dir=ENV_DIR,
            cv_config=ISOTONIC_LOGISTIC_L2_LOG_CPM_FULL
        )
    run:
        commands = [
            'mkdir -p {tmp_dir}/leave_study_out_cv_condor/toy/{cv_config}'.format(
                tmp_dir=TMP_DIR,
                cv_config=ISOTONIC_LOGISTIC_L2_LOG_CPM_FULL
            ),
            'mkdir -p {env_dir}/data/experiment_lists/toy/leave_study_out_cv_results/{cv_config}'.format(
                env_dir=ENV_DIR,
                cv_config=ISOTONIC_LOGISTIC_L2_LOG_CPM_FULL
            ),
            'python2.7 {project_dir}/learning/leave_study_out_cv_condor_primary.py {project_dir}/learning/leave_study_out_cv_config/{env_name}/{cv_config}.json {env_dir} toy -o {{output}} -r {tmp_dir}//leave_study_out_cv_condor/toy/{cv_config}'.format(
                project_dir=PROJECT_DIR,
                env_name=ENV_NAME,
                env_dir=ENV_DIR,
                tmp_dir=TMP_DIR,
                cv_config=ISOTONIC_LOGISTIC_L2_LOG_CPM_FULL
            )
        ]
        for c in commands:
            shell(c)

rule toy_tpr_logistic_full_leave_study_out_cv_condor:
    input:
        data_set='{}/data/data_set_metadata.json'.format(ENV_DIR),
        exp_list_f='{}/data/experiment_lists/toy/experiment_list.json'.format(ENV_DIR),
        labelling_f='{}/data/experiment_lists/toy/labelling.json'.format(ENV_DIR)
    output:
        '{env_dir}/data/experiment_lists/toy/leave_study_out_cv_results/{cv_config}/leave_study_out_cv_results.json'.format(
            env_dir=ENV_DIR,
            cv_config=TPR_LOGISTIC_L2_LOG_CPM_FULL
        )
    run:
        commands = [
            'mkdir -p {tmp_dir}/leave_study_out_cv_condor/toy/{cv_config}'.format(
                tmp_dir=TMP_DIR,
                cv_config=TPR_LOGISTIC_L2_LOG_CPM_FULL
            ),
            'mkdir -p {env_dir}/data/experiment_lists/toy/leave_study_out_cv_results/{cv_config}'.format(
                env_dir=ENV_DIR,
                cv_config=TPR_LOGISTIC_L2_LOG_CPM_FULL
            ),
            'python2.7 {project_dir}/learning/leave_study_out_cv_condor_primary.py {project_dir}/learning/leave_study_out_cv_config/{env_name}/{cv_config}.json {env_dir} toy -o {{output}} -r {tmp_dir}//leave_study_out_cv_condor/toy/{cv_config}'.format(
                project_dir=PROJECT_DIR,
                env_name=ENV_NAME,
                env_dir=ENV_DIR,
                tmp_dir=TMP_DIR,
                cv_config=TPR_LOGISTIC_L2_LOG_CPM_FULL
            )
        ]
        for c in commands:
            shell(c)


rule toy_bnc_leave_study_out_cv:
    input:
        '{}/leave_study_out_cv_batches'.format(TMP_DIR),
        '{}/data/experiment_lists/toy/experiment_list.json'.format(ENV_DIR)
    output:
        # Make sure the experiment configuration name is the output directory!
        '{}/data/experiment_lists/toy/leave_study_out_cv_results/bnc.linear_svm.dynamic_bins_10.bin_pseudo_1.counts_prior.gibbs.burn_in_30000/leave_study_out_cv_results.json'.format(ENV_DIR)
    params:
        exp_config_name = 'bnc.linear_svm.dynamic_bins_10.bin_pseudo_1.counts_prior.gibbs.burn_in_30000'
    run:
        commands = [
            'mkdir -p {}/data/experiment_lists/toy/leave_study_out_cv_results/{{params.exp_config_name}}'.format(ENV_DIR),
            'mkdir -p {}/leave_study_out_cv_batches/toy/{{params.exp_config_name}}'.format(TMP_DIR),
            'mkdir -p {}/leave_study_out_cv_artifacts/toy/{{params.exp_config_name}}'.format(TMP_DIR),
            'python2.7 {project_dir}/learning/leave_study_out_cv.py {project_dir}/learning/leave_study_out_cv_config/{env_name}/{{params.exp_config_name}}.json {env_dir} toy -b {tmp_dir}/leave_study_out_cv_batches/toy/{{params.exp_config_name}} -r {tmp_dir}/leave_study_out_cv_artifacts/toy/{{params.exp_config_name}} -o {{output}}'.format(
                project_dir=PROJECT_DIR,
                env_name=ENV_NAME,
                env_dir=ENV_DIR,
                tmp_dir=TMP_DIR
            )
        ]
        for c in commands:
            shell(c)


rule toy_cdc_svm_rbc_remove_ambig_leave_study_out_cv:
    input:
        '{}/leave_study_out_cv_batches'.format(TMP_DIR),
        '{}/data/experiment_lists/toy/experiment_list.json'.format(ENV_DIR)
    output:
        # Make sure the experiment configuration name is the output directory!
        '{}/data/experiment_lists/toy/leave_study_out_cv_results/cdc.svm.rbf.assert_ambig_neg/leave_study_out_cv_results.json'.format(ENV_DIR)
    params:
        exp_config_name = 'cdc.svm.rbf.assert_ambig_neg'
    run:
        commands = [
            'mkdir -p {}/data/experiment_lists/toy/leave_study_out_cv_results/{{params.exp_config_name}}'.format(ENV_DIR),
            'mkdir -p {}/leave_study_out_cv_batches/toy/{{params.exp_config_name}}'.format(TMP_DIR),
            'mkdir -p {}/leave_study_out_cv_artifacts/toy/{{params.exp_config_name}}'.format(TMP_DIR),
            'python2.7 {project_dir}/learning/leave_study_out_cv.py {project_dir}/learning/leave_study_out_cv_config/{env_name}/{{params.exp_config_name}}.json {env_dir} toy -b {tmp_dir}/leave_study_out_cv_batches/toy/{{params.exp_config_name}} -r {tmp_dir}/leave_study_out_cv_artifacts/toy/{{params.exp_config_name}} -o {{output}}'.format(
                project_dir=PROJECT_DIR,
                env_name=ENV_NAME,
                env_dir=ENV_DIR,
                tmp_dir=TMP_DIR
            )
        ]
        for c in commands:
            shell(c)


rule test_cdc_assert_ambig_neg_downweight_by_group_leave_study_out_cv:
    input:
        '{}/leave_study_out_cv_batches'.format(TMP_DIR),
        '{}/data/experiment_lists/test_experiments/experiment_list.json'.format(ENV_DIR)
    output:
        # Make sure the experiment configuration name is the output directory!
        '{}/data/experiment_lists/test_experiments/leave_study_out_cv_results/cdc.logistic_regression.l2.assert_ambig_neg.downweight_by_group.collapse_ontology/leave_study_out_cv_results.json'.format(ENV_DIR)
    params:
        exp_config_name = 'cdc.logistic_regression.l2.assert_ambig_neg.downweight_by_group.collapse_ontology'
    run:
        commands = [
            'mkdir -p {}/data/experiment_lists/test_experiments/leave_study_out_cv_results/{{params.exp_config_name}}'.format(ENV_DIR),
            'mkdir -p {}/leave_study_out_cv_batches/test_experiments/{{params.exp_config_name}}'.format(TMP_DIR),
            'mkdir -p {}/leave_study_out_cv_artifacts/test_experiments/{{params.exp_config_name}}'.format(TMP_DIR),
            'python2.7 {project_dir}/learning/leave_study_out_cv.py {project_dir}/learning/leave_study_out_cv_config/{env_name}/{{params.exp_config_name}}.json {env_dir} test_experiments -b {tmp_dir}/leave_study_out_cv_batches/test_experiments/{{params.exp_config_name}} -r {tmp_dir}/leave_study_out_cv_artifacts/test_experiments/{{params.exp_config_name}} -o {{output}}'.format(
                project_dir=PROJECT_DIR,
                env_name=ENV_NAME,
                env_dir=ENV_DIR,
                tmp_dir=TMP_DIR
            )
        ]
        for c in commands:
            shell(c)
    
rule test_cdc_ambig_latent_leave_study_out_cv:
    input:
        '{}/leave_study_out_cv_batches'.format(TMP_DIR),
        '{}/data/experiment_lists/test_experiments/experiment_list.json'.format(ENV_DIR)
    output:
        # Make sure the experiment configuration name is the output directory!
        '{}/data/experiment_lists/test_experiments/leave_study_out_cv_results/cdc.logistic_regression.l2.ambig_latent.collapse_ontology/leave_study_out_cv_results.json'.format(ENV_DIR)
    params:
        exp_config_name = 'cdc.logistic_regression.l2.ambig_latent.collapse_ontology'
    run:
        commands = [
            'mkdir -p {}/data/experiment_lists/test_experiments/leave_study_out_cv_results/{{params.exp_config_name}}'.format(ENV_DIR),
            'mkdir -p {}/leave_study_out_cv_batches/test_experiments/{{params.exp_config_name}}'.format(TMP_DIR),
            'mkdir -p {}/leave_study_out_cv_artifacts/test_experiments/{{params.exp_config_name}}'.format(TMP_DIR),
            'python2.7 {project_dir}/learning/leave_study_out_cv.py {project_dir}/learning/leave_study_out_cv_config/{env_name}/{{params.exp_config_name}}.json {env_dir} test_experiments -b {tmp_dir}/leave_study_out_cv_batches/test_experiments/{{params.exp_config_name}} -r {tmp_dir}/leave_study_out_cv_artifacts/test_experiments/{{params.exp_config_name}} -o {{output}}'.format(
                project_dir=PROJECT_DIR,
                env_name=ENV_NAME,
                env_dir=ENV_DIR,
                tmp_dir=TMP_DIR
            )
        ]
        for c in commands:
            shell(c)


rule toy_cdc_leave_study_out_cv:
    input:
        '{}/leave_study_out_cv_batches'.format(TMP_DIR),
        '{}/data/experiment_lists/toy/experiment_list.json'.format(ENV_DIR)
    output:
        # Make sure the experiment configuration name is the output directory!
        '{}/data/experiment_lists/toy/leave_study_out_cv_results/cdc.logistic_regression.l2.remove_ambig/leave_study_out_cv_results.json'.format(ENV_DIR)
    params:
        exp_config_name = 'cdc.logistic_regression.l2.remove_ambig'
    run:
        commands = [
            'mkdir -p {}/data/experiment_lists/toy/leave_study_out_cv_results/{{params.exp_config_name}}'.format(ENV_DIR),
            'mkdir -p {}/leave_study_out_cv_batches/toy/{{params.exp_config_name}}'.format(TMP_DIR),
            'mkdir -p {}/leave_study_out_cv_artifacts/toy/{{params.exp_config_name}}'.format(TMP_DIR),
            'python2.7 {project_dir}/learning/leave_study_out_cv.py {project_dir}/learning/leave_study_out_cv_config/{env_name}/{{params.exp_config_name}}.json {env_dir} toy -b {tmp_dir}/leave_study_out_cv_batches/toy/{{params.exp_config_name}} -r {tmp_dir}/leave_study_out_cv_artifacts/toy/{{params.exp_config_name}} -o {{output}}'.format(
                project_dir=PROJECT_DIR,
                env_name=ENV_NAME,
                env_dir=ENV_DIR,
                tmp_dir=TMP_DIR
            )
        ]
        for c in commands:
            shell(c)


rule test_cdc_leave_study_out_cv:
    input:
        '{}/leave_study_out_cv_batches'.format(TMP_DIR),
        '{}/data/experiment_lists/test_experiments/experiment_list.json'.format(ENV_DIR)
    output:
        # Make sure the experiment configuration name is the output directory!
        '{}/data/experiment_lists/test_experiments/leave_study_out_cv_results/cdc.logistic_regression.l2.remove_ambig/leave_study_out_cv_results.json'.format(ENV_DIR)
    params:
        exp_config_name = 'cdc.logistic_regression.l2.remove_ambig'
    run:
        commands = [
            'mkdir -p {}/data/experiment_lists/test_experiments/leave_study_out_cv_results/{{params.exp_config_name}}'.format(ENV_DIR),
            'mkdir -p {}/leave_study_out_cv_batches/{{params.exp_config_name}}'.format(TMP_DIR),
            'mkdir -p {}/leave_study_out_cv_artifacts/{{params.exp_config_name}}'.format(TMP_DIR),
            'python2.7 {project_dir}/learning/leave_study_out_cv.py {project_dir}/learning/leave_study_out_cv_config/{env_name}/{{params.exp_config_name}}.json {env_dir} test_experiments -b {tmp_dir}/leave_study_out_cv_batches/{{params.exp_config_name}} -r {tmp_dir}/leave_study_out_cv_artifacts/{{params.exp_config_name}} -o {{output}}'.format(
                project_dir=PROJECT_DIR,
                env_name=ENV_NAME,
                env_dir=ENV_DIR,
                tmp_dir=TMP_DIR
            )
        ]
        for c in commands:
            shell(c)


rule test_bnc_leave_study_out_cv:
    input:
        '{}/leave_study_out_cv_batches'.format(TMP_DIR),
        '{}/data/experiment_lists/test_experiments/experiment_list.json'.format(ENV_DIR)
    output:
        '{}/data/experiment_lists/test_experiments/leave_study_out_cv_results/leave_study_out_cv_results.json'.format(ENV_DIR)
    params:
        exp_config_name = 'bnc.linear_svm.dynamic_bins_10.bin_pseudo_1.counts_prior'
    run:
        commands = [
            'mkdir -p {}/leave_study_out_cv_batches/{{params.exp_config_name}}'.format(TMP_DIR),
            'mkdir -p {}/leave_study_out_cv_artifacts/{{params.exp_config_name}}'.format(TMP_DIR),
            'python2.7 {project_dir}/learning/leave_study_out_cv.py {project_dir}/learning/leave_study_out_cv_config/{env_name}/{{params.exp_config_name}}.json {env_dir} test_experiments -b {tmp_dir}/leave_study_out_cv_batches/{{params.exp_config_name}} -r {tmp_dir}/leave_study_out_cv_artifacts/{{params.exp_config_name}} -o {{output}}'.format(
                project_dir=PROJECT_DIR,
                env_name=ENV_NAME,
                env_dir=ENV_DIR,
                tmp_dir=TMP_DIR
            )
        ]
        for c in commands:
            shell(c)


rule test_one_nn_leave_study_out_cv:
    input:
        '{}/leave_study_out_cv_batches'.format(TMP_DIR),
        '{}/data/experiment_lists/test_experiments/experiment_list.json'.format(ENV_DIR)
    output:
        '{env_dir}/data/experiment_lists/test_experiments/{exp_config_name}/leave_study_out_cv_results/leave_study_out_cv_results.json'.format(
            env_dir=ENV_DIR,
            exp_config_name=ONE_NN_LOG_CPM_CORRELATION
        )
    run:
        commands = [
            'mkdir {tmp_dir}/leave_study_out_cv_batches/{exp_config_name}'.format(
                tmp_dir=TMP_DIR,
                exp_config_name=ONE_NN_LOG_CPM_CORRELATION
            ),
            'python2.7 {project_dir}/learning/leave_study_out_cv.py {project_dir}/learning/leave_study_out_cv_config/{env_name}/{exp_config_name}.json {env_dir} test_experiments -b {tmp_dir}/leave_study_out_cv_batches/{exp_config_name}  -o {{output}}'.format(
                project_dir=PROJECT_DIR,
                env_name=ENV_NAME,
                env_dir=ENV_DIR,
                tmp_dir=TMP_DIR,
                exp_config_name=ONE_NN_LOG_CPM_CORRELATION
            )
        ]
        for c in commands:
            shell(c)


#####################################################################
#   Train models on the training set
####################################################################

rule train_per_label_logistic_regression_l2_assert_ambig_neg:
    input:
        '{env_dir}/data/experiment_lists/{exp_list}/experiment_list.json'.format(
            env_dir=ENV_DIR,
            exp_list='train_set.untampered_bulk_primary_cells_with_data'
        ),
        '{env_dir}/data/experiment_lists/{exp_list}/labelling.json'.format(
            env_dir=ENV_DIR,
            exp_list='train_set.untampered_bulk_primary_cells_with_data'
        )
    output:
        '{env_dir}/data/experiment_lists/{exp_list}/trained_models/{algo_config}/model.pickle'.format(
            env_dir=ENV_DIR,
            exp_list='train_set.untampered_bulk_primary_cells_with_data',
            algo_config=PER_LABEL_LOG_CPM_LOGISTIC_L2_FULL
        )
    run:
        commands = [
            'mkdir -p /scratch/mnbernstein/cell_type_phenotyping_tmp/train_and_save_artifacts',
            'mkdir -p {env_dir}/data/experiment_lists/{exp_list}/trained_models/{algo_config}'.format(
                env_dir=ENV_DIR,
                exp_list='train_set.untampered_bulk_primary_cells_with_data',
                algo_config=PER_LABEL_LOG_CPM_LOGISTIC_L2_FULL
            ),
            'python2.7 {project_dir}/learning/train_and_save.py {project_dir}/learning/train_and_save_config/{env_name}/{algo_config}.json {env_dir} {exp_list} -o {env_dir}/data/experiment_lists/{exp_list}/trained_models/{algo_config} -r /scratch/mnbernstein/cell_type_phenotyping_tmp/train_and_save_artifacts'.format(
                env_dir=ENV_DIR,
                env_name=ENV_NAME,
                exp_list='train_set.untampered_bulk_primary_cells_with_data',
                algo_config=PER_LABEL_LOG_CPM_LOGISTIC_L2_FULL,
                project_dir=PROJECT_DIR
            )
        ]
        for c in commands:
            shell(c)

rule train_per_label_logistic_regression_l2_assert_ambig_neg_downweight:
    input:
        '{env_dir}/data/experiment_lists/{exp_list}/experiment_list.json'.format(
            env_dir=ENV_DIR,
            exp_list='train_set.untampered_bulk_primary_cells_with_data'
        ),
        '{env_dir}/data/experiment_lists/{exp_list}/labelling.json'.format(
            env_dir=ENV_DIR,
            exp_list='train_set.untampered_bulk_primary_cells_with_data'
        )
    output:
        '{env_dir}/data/experiment_lists/{exp_list}/trained_models/{algo_config}/model.pickle'.format(
            env_dir=ENV_DIR,
            exp_list='train_set.untampered_bulk_primary_cells_with_data',
            algo_config=PER_LABEL_LOG_CPM_LOGISTIC_L2_FULL_DOWNWEIGHT
        )
    run:
        commands = [
            'mkdir -p /scratch/mnbernstein/cell_type_phenotyping_tmp/train_and_save_artifacts',
            'mkdir -p {env_dir}/data/experiment_lists/{exp_list}/trained_models/{algo_config}'.format(
                env_dir=ENV_DIR,
                exp_list='train_set.untampered_bulk_primary_cells_with_data',
                algo_config=PER_LABEL_LOG_CPM_LOGISTIC_L2_FULL_DOWNWEIGHT
            ),
            'python2.7 {project_dir}/learning/train_and_save.py {project_dir}/learning/train_and_save_config/{env_name}/{algo_config}.json {env_dir} {exp_list} -o {env_dir}/data/experiment_lists/{exp_list}/trained_models/{algo_config} -r /scratch/mnbernstein/cell_type_phenotyping_tmp/train_and_save_artifacts'.format(
                env_dir=ENV_DIR,
                env_name=ENV_NAME,
                exp_list='train_set.untampered_bulk_primary_cells_with_data',
                algo_config=PER_LABEL_LOG_CPM_LOGISTIC_L2_FULL_DOWNWEIGHT,
                project_dir=PROJECT_DIR
            )
        ]
        for c in commands:
            shell(c)


rule train_cdc_logistic_regression_l2_assert_ambig_neg_downweight_by_group:
    input:
        '{env_dir}/data/experiment_lists/{exp_list}/experiment_list.json'.format(
            env_dir=ENV_DIR,
            exp_list='train_set.untampered_bulk_primary_cells_with_data'
        ),
        '{env_dir}/data/experiment_lists/{exp_list}/labelling.json'.format(
            env_dir=ENV_DIR, 
            exp_list='train_set.untampered_bulk_primary_cells_with_data'
        )
    output:
        '{env_dir}/data/experiment_lists/{exp_list}/trained_models/{algo_config}/model.pickle'.format(
            env_dir=ENV_DIR,
            exp_list='train_set.untampered_bulk_primary_cells_with_data',
            algo_config=CDC_LOG_CPM_LOGISTIC_L2_ASSERT_AMBIG_NEG_FULL_DOWNWEIGHT
        )
    run:
        commands = [
            'mkdir -p /scratch/mnbernstein/cell_type_phenotyping_tmp/train_and_save_artifacts',
            'mkdir -p {env_dir}/data/experiment_lists/{exp_list}/trained_models/{algo_config}'.format(
                env_dir=ENV_DIR,
                exp_list='train_set.untampered_bulk_primary_cells_with_data',
                algo_config=CDC_LOG_CPM_LOGISTIC_L2_ASSERT_AMBIG_NEG_FULL_DOWNWEIGHT
            ),
            'python2.7 {project_dir}/learning/train_and_save.py {project_dir}/learning/train_and_save_config/{env_name}/{algo_config}.json {env_dir} {exp_list} -o {env_dir}/data/experiment_lists/{exp_list}/trained_models/{algo_config} -r /scratch/mnbernstein/cell_type_phenotyping_tmp/train_and_save_artifacts'.format(
                env_dir=ENV_DIR,
                env_name=ENV_NAME,
                exp_list='train_set.untampered_bulk_primary_cells_with_data',
                algo_config=CDC_LOG_CPM_LOGISTIC_L2_ASSERT_AMBIG_NEG_FULL_DOWNWEIGHT,
                project_dir=PROJECT_DIR
            )
        ]
        for c in commands:
            shell(c)

rule train_full_data_cdc_logistic_regression_l2_assert_ambig_neg_downweight_by_group:
    input:
        '{env_dir}/data/experiment_lists/{exp_list}/experiment_list.json'.format(
            env_dir=ENV_DIR,
            exp_list='untampered_bulk_primary_cells_with_data'
        ),
        '{env_dir}/data/experiment_lists/{exp_list}/labelling.json'.format(
            env_dir=ENV_DIR,
            exp_list='untampered_bulk_primary_cells_with_data'
        )
    output:
        '{env_dir}/data/experiment_lists/{exp_list}/trained_models/{algo_config}/model.pickle'.format(
            env_dir=ENV_DIR,
            exp_list='untampered_bulk_primary_cells_with_data',
            algo_config=CDC_LOG_CPM_LOGISTIC_L2_ASSERT_AMBIG_NEG_FULL_DOWNWEIGHT
        )
    run:
        commands = [
            'mkdir -p /scratch/mnbernstein/cell_type_phenotyping_tmp/train_and_save_artifacts',
            'mkdir -p {env_dir}/data/experiment_lists/{exp_list}/trained_models/{algo_config}'.format(
                env_dir=ENV_DIR,
                exp_list='untampered_bulk_primary_cells_with_data',
                algo_config=CDC_LOG_CPM_LOGISTIC_L2_ASSERT_AMBIG_NEG_FULL_DOWNWEIGHT
            ),
            'python2.7 {project_dir}/learning/train_and_save.py {project_dir}/learning/train_and_save_config/{env_name}/{algo_config}.json {env_dir} {exp_list} -o {env_dir}/data/experiment_lists/{exp_list}/trained_models/{algo_config} -r /scratch/mnbernstein/cell_type_phenotyping_tmp/train_and_save_artifacts'.format(
                env_dir=ENV_DIR,
                env_name=ENV_NAME,
                exp_list='untampered_bulk_primary_cells_with_data',
                algo_config=CDC_LOG_CPM_LOGISTIC_L2_ASSERT_AMBIG_NEG_FULL_DOWNWEIGHT,
                project_dir=PROJECT_DIR
            )
        ]
        for c in commands:
            shell(c)

rule toy_train_cdc_logistic_regression_l2_assert_ambig_neg:
    input:
        '{env_dir}/data/experiment_lists/{exp_list}/experiment_list.json'.format(
            env_dir=ENV_DIR,
            exp_list='toy'
        ),
        '{env_dir}/data/experiment_lists/{exp_list}/labelling.json'.format(
            env_dir=ENV_DIR,
            exp_list='toy'
        )
    output:
        '{env_dir}/data/experiment_lists/{exp_list}/trained_models/{algo_config}/model.pickle'.format(
            env_dir=ENV_DIR,
            exp_list='toy',
            algo_config=CDC_LOG_CPM_LOGISTIC_L2_ASSERT_AMBIG_NEG_FULL
        )
    run:
        commands = [
            'mkdir -p /scratch/mnbernstein/cell_type_phenotyping_tmp/train_and_save_artifacts',
            'mkdir -p {env_dir}/data/experiment_lists/{exp_list}/trained_models/{algo_config}'.format(
                env_dir=ENV_DIR,
                exp_list='toy',
                algo_config=CDC_LOG_CPM_LOGISTIC_L2_ASSERT_AMBIG_NEG_FULL
            ),
            'python2.7 {project_dir}/learning/train_and_save.py {project_dir}/learning/train_and_save_config/{env_name}/{algo_config}.json {env_dir} {exp_list} -o {env_dir}/data/experiment_lists/{exp_list}/trained_models/{algo_config} -r /scratch/mnbernstein/cell_type_phenotyping_tmp/train_and_save_artifacts'.format(
                env_dir=ENV_DIR,
                env_name=ENV_NAME,
                exp_list='toy',
                algo_config=CDC_LOG_CPM_LOGISTIC_L2_ASSERT_AMBIG_NEG_FULL,
                project_dir=PROJECT_DIR
            )
        ]
        for c in commands:
            shell(c)

rule train_cdc_logistic_regression_l2_assert_ambig_neg:
    input:
        '{env_dir}/data/experiment_lists/{exp_list}/experiment_list.json'.format(
            env_dir=ENV_DIR,
            exp_list='train_set.untampered_bulk_primary_cells_with_data'
        ),
        '{env_dir}/data/experiment_lists/{exp_list}/labelling.json'.format(
            env_dir=ENV_DIR,
            exp_list='train_set.untampered_bulk_primary_cells_with_data'
        )
    output:
        '{env_dir}/data/experiment_lists/{exp_list}/trained_models/{algo_config}/model.pickle'.format(
            env_dir=ENV_DIR,
            exp_list='train_set.untampered_bulk_primary_cells_with_data',
            algo_config=CDC_LOG_CPM_LOGISTIC_L2_ASSERT_AMBIG_NEG_FULL
        )
    run:
        commands = [
            'mkdir -p /scratch/mnbernstein/cell_type_phenotyping_tmp/train_and_save_artifacts',
            'mkdir -p {env_dir}/data/experiment_lists/{exp_list}/trained_models/{algo_config}'.format(
                env_dir=ENV_DIR,
                exp_list='train_set.untampered_bulk_primary_cells_with_data',
                algo_config=CDC_LOG_CPM_LOGISTIC_L2_ASSERT_AMBIG_NEG_FULL
            ),
            'python2.7 {project_dir}/learning/train_and_save.py {project_dir}/learning/train_and_save_config/{env_name}/{algo_config}.json {env_dir} {exp_list} -o {env_dir}/data/experiment_lists/{exp_list}/trained_models/{algo_config} -r /scratch/mnbernstein/cell_type_phenotyping_tmp/train_and_save_artifacts'.format(
                env_dir=ENV_DIR,
                env_name=ENV_NAME,
                exp_list='train_set.untampered_bulk_primary_cells_with_data',
                algo_config=CDC_LOG_CPM_LOGISTIC_L2_ASSERT_AMBIG_NEG_FULL,
                project_dir=PROJECT_DIR
            )
        ]
        for c in commands:
            shell(c)    


rule train_full_data_cdc_logistic_regression_l2_assert_ambig_neg:
    input:
        '{env_dir}/data/experiment_lists/{exp_list}/experiment_list.json'.format(
            env_dir=ENV_DIR,
            exp_list='untampered_bulk_primary_cells_with_data'
        ),
        '{env_dir}/data/experiment_lists/{exp_list}/labelling.json'.format(
            env_dir=ENV_DIR,
            exp_list='untampered_bulk_primary_cells_with_data'
        )
    output:
        '{env_dir}/data/experiment_lists/{exp_list}/trained_models/{algo_config}/model.pickle'.format(
            env_dir=ENV_DIR,
            exp_list='untampered_bulk_primary_cells_with_data',
            algo_config=CDC_LOG_CPM_LOGISTIC_L2_ASSERT_AMBIG_NEG_FULL
        )
    run:
        commands = [
            'mkdir -p /scratch/mnbernstein/cell_type_phenotyping_tmp/train_and_save_artifacts',
            'mkdir -p {env_dir}/data/experiment_lists/{exp_list}/trained_models/{algo_config}'.format(
                env_dir=ENV_DIR,
                exp_list='untampered_bulk_primary_cells_with_data',
                algo_config=CDC_LOG_CPM_LOGISTIC_L2_ASSERT_AMBIG_NEG_FULL
            ),
            'python2.7 {project_dir}/learning/train_and_save.py {project_dir}/learning/train_and_save_config/{env_name}/{algo_config}.json {env_dir} {exp_list} -o {env_dir}/data/experiment_lists/{exp_list}/trained_models/{algo_config} -r /scratch/mnbernstein/cell_type_phenotyping_tmp/train_and_save_artifacts'.format(
                env_dir=ENV_DIR,
                env_name=ENV_NAME,
                exp_list='untampered_bulk_primary_cells_with_data',
                algo_config=CDC_LOG_CPM_LOGISTIC_L2_ASSERT_AMBIG_NEG_FULL,
                project_dir=PROJECT_DIR
            )
        ]
        for c in commands:
            shell(c)

rule train_isotonic_logistic_regression_l2_assert_ambig_neg:
    input:
        '{env_dir}/data/experiment_lists/{exp_list}/experiment_list.json'.format(
            env_dir=ENV_DIR,
            exp_list='train_set.untampered_bulk_primary_cells_with_data'
        ),
        '{env_dir}/data/experiment_lists/{exp_list}/labelling.json'.format(
            env_dir=ENV_DIR,
            exp_list='train_set.untampered_bulk_primary_cells_with_data'
        )
    output:
        '{env_dir}/data/experiment_lists/{exp_list}/trained_models/{algo_config}/model.pickle'.format(
            env_dir=ENV_DIR,
            exp_list='train_set.untampered_bulk_primary_cells_with_data',
            algo_config=ISOTONIC_LOGISTIC_L2_LOG_CPM_ASSERT_AMBIG_NEG_FULL
        )
    run:
        commands = [
            'mkdir -p /scratch/mnbernstein/cell_type_phenotyping_tmp/train_and_save_artifacts',
            'mkdir -p {env_dir}/data/experiment_lists/{exp_list}/trained_models/{algo_config}'.format(
                env_dir=ENV_DIR,
                exp_list='train_set.untampered_bulk_primary_cells_with_data',
                algo_config=ISOTONIC_LOGISTIC_L2_LOG_CPM_ASSERT_AMBIG_NEG_FULL
            ),
            'python2.7 {project_dir}/learning/train_and_save.py {project_dir}/learning/train_and_save_config/{env_name}/{algo_config}.json {env_dir} {exp_list} -o {env_dir}/data/experiment_lists/{exp_list}/trained_models/{algo_config} -r /scratch/mnbernstein/cell_type_phenotyping_tmp/train_and_save_artifacts'.format(
                env_dir=ENV_DIR,
                env_name=ENV_NAME,
                exp_list='train_set.untampered_bulk_primary_cells_with_data',
                algo_config=ISOTONIC_LOGISTIC_L2_LOG_CPM_ASSERT_AMBIG_NEG_FULL,
                project_dir=PROJECT_DIR
            )
        ]
        for c in commands:
            shell(c)

rule train_full_data_isotonic_logistic_regression_l2_assert_ambig_neg:
    input:
        '{env_dir}/data/experiment_lists/{exp_list}/experiment_list.json'.format(
            env_dir=ENV_DIR,
            exp_list='untampered_bulk_primary_cells_with_data'
        ),
        '{env_dir}/data/experiment_lists/{exp_list}/labelling.json'.format(
            env_dir=ENV_DIR,
            exp_list='untampered_bulk_primary_cells_with_data'
        )
    output:
        '{env_dir}/data/experiment_lists/{exp_list}/trained_models/{algo_config}/model.pickle'.format(
            env_dir=ENV_DIR,
            exp_list='untampered_bulk_primary_cells_with_data',
            algo_config=ISOTONIC_LOGISTIC_L2_LOG_CPM_ASSERT_AMBIG_NEG_FULL
        )
    run:
        commands = [
            'mkdir -p /scratch/mnbernstein/cell_type_phenotyping_tmp/train_and_save_artifacts',
            'mkdir -p {env_dir}/data/experiment_lists/{exp_list}/trained_models/{algo_config}'.format(
                env_dir=ENV_DIR,
                exp_list='untampered_bulk_primary_cells_with_data',
                algo_config=ISOTONIC_LOGISTIC_L2_LOG_CPM_ASSERT_AMBIG_NEG_FULL
            ),
            'python2.7 {project_dir}/learning/train_and_save.py {project_dir}/learning/train_and_save_config/{env_name}/{algo_config}.json {env_dir} {exp_list} -o {env_dir}/data/experiment_lists/{exp_list}/trained_models/{algo_config} -r /scratch/mnbernstein/cell_type_phenotyping_tmp/train_and_save_artifacts'.format(
                env_dir=ENV_DIR,
                env_name=ENV_NAME,
                exp_list='untampered_bulk_primary_cells_with_data',
                algo_config=ISOTONIC_LOGISTIC_L2_LOG_CPM_ASSERT_AMBIG_NEG_FULL,
                project_dir=PROJECT_DIR
            )
        ]
        for c in commands:
            shell(c)

rule train_isotonic_logistic_regression_l2_assert_ambig_neg_downweight_by_group:
    input:
        '{env_dir}/data/experiment_lists/{exp_list}/experiment_list.json'.format(
            env_dir=ENV_DIR,
            exp_list='train_set.untampered_bulk_primary_cells_with_data'
        ),
        '{env_dir}/data/experiment_lists/{exp_list}/labelling.json'.format(
            env_dir=ENV_DIR,
            exp_list='train_set.untampered_bulk_primary_cells_with_data'
        )
    output:
        '{env_dir}/data/experiment_lists/{exp_list}/trained_models/{algo_config}/model.pickle'.format(
            env_dir=ENV_DIR,
            exp_list='train_set.untampered_bulk_primary_cells_with_data',
            algo_config=ISOTONIC_LOGISTIC_L2_LOG_CPM_ASSERT_AMBIG_NEG_FULL_DOWNWEIGHT
        )
    run:
        commands = [
            'mkdir -p /scratch/mnbernstein/cell_type_phenotyping_tmp/train_and_save_artifacts',
            'mkdir -p {env_dir}/data/experiment_lists/{exp_list}/trained_models/{algo_config}'.format(
                env_dir=ENV_DIR,
                exp_list='train_set.untampered_bulk_primary_cells_with_data',
                algo_config=ISOTONIC_LOGISTIC_L2_LOG_CPM_ASSERT_AMBIG_NEG_FULL_DOWNWEIGHT
            ),
            'python2.7 {project_dir}/learning/train_and_save.py {project_dir}/learning/train_and_save_config/{env_name}/{algo_config}.json {env_dir} {exp_list} -o {env_dir}/data/experiment_lists/{exp_list}/trained_models/{algo_config} -r /scratch/mnbernstein/cell_type_phenotyping_tmp/train_and_save_artifacts'.format(
                env_dir=ENV_DIR,
                env_name=ENV_NAME,
                exp_list='train_set.untampered_bulk_primary_cells_with_data',
                algo_config=ISOTONIC_LOGISTIC_L2_LOG_CPM_ASSERT_AMBIG_NEG_FULL_DOWNWEIGHT,
                project_dir=PROJECT_DIR
            )
        ]
        for c in commands:
            shell(c)

rule train_full_data_isotonic_logistic_regression_l2_assert_ambig_neg_downweight_by_group:
    input:
        '{env_dir}/data/experiment_lists/{exp_list}/experiment_list.json'.format(
            env_dir=ENV_DIR,
            exp_list='untampered_bulk_primary_cells_with_data'
        ),
        '{env_dir}/data/experiment_lists/{exp_list}/labelling.json'.format(
            env_dir=ENV_DIR,
            exp_list='untampered_bulk_primary_cells_with_data'
        )
    output:
        '{env_dir}/data/experiment_lists/{exp_list}/trained_models/{algo_config}/model.pickle'.format(
            env_dir=ENV_DIR,
            exp_list='untampered_bulk_primary_cells_with_data',
            algo_config=ISOTONIC_LOGISTIC_L2_LOG_CPM_ASSERT_AMBIG_NEG_FULL_DOWNWEIGHT
        )
    run:
        commands = [
            'mkdir -p /scratch/mnbernstein/cell_type_phenotyping_tmp/train_and_save_artifacts',
            'mkdir -p {env_dir}/data/experiment_lists/{exp_list}/trained_models/{algo_config}'.format(
                env_dir=ENV_DIR,
                exp_list='untampered_bulk_primary_cells_with_data',
                algo_config=ISOTONIC_LOGISTIC_L2_LOG_CPM_ASSERT_AMBIG_NEG_FULL_DOWNWEIGHT
            ),
            'python2.7 {project_dir}/learning/train_and_save.py {project_dir}/learning/train_and_save_config/{env_name}/{algo_config}.json {env_dir} {exp_list} -o {env_dir}/data/experiment_lists/{exp_list}/trained_models/{algo_config} -r /scratch/mnbernstein/cell_type_phenotyping_tmp/train_and_save_artifacts'.format(
                env_dir=ENV_DIR,
                env_name=ENV_NAME,
                exp_list='untampered_bulk_primary_cells_with_data',
                algo_config=ISOTONIC_LOGISTIC_L2_LOG_CPM_ASSERT_AMBIG_NEG_FULL_DOWNWEIGHT,
                project_dir=PROJECT_DIR
            )
        ]
        for c in commands:
            shell(c)


rule train_bnc_linear_svm_count_cond_prior:
    input:
        '{env_dir}/data/experiment_lists/{exp_list}/experiment_list.json'.format(
            env_dir=ENV_DIR,
            exp_list='train_set.untampered_bulk_primary_cells_with_data'
        ),
        '{env_dir}/data/experiment_lists/{exp_list}/labelling.json'.format(
            env_dir=ENV_DIR,
            exp_list='train_set.untampered_bulk_primary_cells_with_data'
        )
    output:
        '{env_dir}/data/experiment_lists/{exp_list}/trained_models/{algo_config}/model.pickle'.format(
            env_dir=ENV_DIR,
            exp_list='train_set.untampered_bulk_primary_cells_with_data',
            algo_config=BNC_LOG_CPM_LINEAR_SVM_COND
        )
    run:
        commands = [
            'mkdir -p /scratch/mnbernstein/cell_type_phenotyping_tmp/train_and_save_artifacts',
            'mkdir -p {env_dir}/data/experiment_lists/{exp_list}/trained_models/{algo_config}'.format(
                env_dir=ENV_DIR,
                exp_list='train_set.untampered_bulk_primary_cells_with_data',
                algo_config=BNC_LOG_CPM_LINEAR_SVM_COND
            ),
            'python2.7 {project_dir}/learning/train_and_save.py {project_dir}/learning/train_and_save_config/{env_name}/{algo_config}.json {env_dir} {exp_list} -o {env_dir}/data/experiment_lists/{exp_list}/trained_models/{algo_config} -r /scratch/mnbernstein/cell_type_phenotyping_tmp/train_and_save_artifacts'.format(
                env_dir=ENV_DIR,
                env_name=ENV_NAME,
                exp_list='train_set.untampered_bulk_primary_cells_with_data',
                algo_config=BNC_LOG_CPM_LINEAR_SVM_COND,
                project_dir=PROJECT_DIR
            )
        ]
        for c in commands:
            shell(c)

rule train_bnc_linear_svm_count_cond_prior_assert_ambig_neg:
    input:
        '{env_dir}/data/experiment_lists/{exp_list}/experiment_list.json'.format(
            env_dir=ENV_DIR,
            exp_list='train_set.untampered_bulk_primary_cells_with_data'
        ),
        '{env_dir}/data/experiment_lists/{exp_list}/labelling.json'.format(
            env_dir=ENV_DIR,
            exp_list='train_set.untampered_bulk_primary_cells_with_data'
        )
    output:
        '{env_dir}/data/experiment_lists/{exp_list}/trained_models/{algo_config}/model.pickle'.format(
            env_dir=ENV_DIR,
            exp_list='train_set.untampered_bulk_primary_cells_with_data',
            algo_config=BNC_LOG_CPM_LINEAR_SVM_COND_ASSERT_AMBIG_NEG
        )
    run:
        commands = [
            'mkdir -p /scratch/mnbernstein/cell_type_phenotyping_tmp/train_and_save_artifacts',
            'mkdir -p {env_dir}/data/experiment_lists/{exp_list}/trained_models/{algo_config}'.format(
                env_dir=ENV_DIR,
                exp_list='train_set.untampered_bulk_primary_cells_with_data',
                algo_config=BNC_LOG_CPM_LINEAR_SVM_COND_ASSERT_AMBIG_NEG
            ),
            'python2.7 {project_dir}/learning/train_and_save.py {project_dir}/learning/train_and_save_config/{env_name}/{algo_config}.json {env_dir} {exp_list} -o {env_dir}/data/experiment_lists/{exp_list}/trained_models/{algo_config} -r /scratch/mnbernstein/cell_type_phenotyping_tmp/train_and_save_artifacts'.format(
                env_dir=ENV_DIR,
                env_name=ENV_NAME,
                exp_list='train_set.untampered_bulk_primary_cells_with_data',
                algo_config=BNC_LOG_CPM_LINEAR_SVM_COND_ASSERT_AMBIG_NEG,
                project_dir=PROJECT_DIR
            )
        ]
        for c in commands:
            shell(c)

rule train_naive_bayes_linear_svm_count_cond_prior_assert_ambig_neg:
    input:
        '{env_dir}/data/experiment_lists/{exp_list}/experiment_list.json'.format(
            env_dir=ENV_DIR,
            exp_list='train_set.untampered_bulk_primary_cells_with_data'
        ),
        '{env_dir}/data/experiment_lists/{exp_list}/labelling.json'.format(
            env_dir=ENV_DIR,
            exp_list='train_set.untampered_bulk_primary_cells_with_data'
        )
    output:
        '{env_dir}/data/experiment_lists/{exp_list}/trained_models/{algo_config}/model.pickle'.format(
            env_dir=ENV_DIR,
            exp_list='train_set.untampered_bulk_primary_cells_with_data',
            algo_config=NAIVE_BAYES_SVM_COND_ASSERT_AMBIG_NEG
        )
    run:
        commands = [
            'mkdir -p /scratch/mnbernstein/cell_type_phenotyping_tmp/train_and_save_artifacts',
            'mkdir -p {env_dir}/data/experiment_lists/{exp_list}/trained_models/{algo_config}'.format(
                env_dir=ENV_DIR,
                exp_list='train_set.untampered_bulk_primary_cells_with_data',
                algo_config=NAIVE_BAYES_SVM_COND_ASSERT_AMBIG_NEG
            ),
            'python2.7 {project_dir}/learning/train_and_save.py {project_dir}/learning/train_and_save_config/{env_name}/{algo_config}.json {env_dir} {exp_list} -o {env_dir}/data/experiment_lists/{exp_list}/trained_models/{algo_config} -r /scratch/mnbernstein/cell_type_phenotyping_tmp/train_and_save_artifacts'.format(
                env_dir=ENV_DIR,
                env_name=ENV_NAME,
                exp_list='train_set.untampered_bulk_primary_cells_with_data',
                algo_config=NAIVE_BAYES_SVM_COND_ASSERT_AMBIG_NEG,
                project_dir=PROJECT_DIR
            )
        ]
        for c in commands:
            shell(c)

rule train_bnc_linear_svm_normal_count_cond_prior:
    input:
        '{env_dir}/data/experiment_lists/{exp_list}/experiment_list.json'.format(
            env_dir=ENV_DIR,
            exp_list='train_set.untampered_bulk_primary_cells_with_data'
        ),
        '{env_dir}/data/experiment_lists/{exp_list}/labelling.json'.format(
            env_dir=ENV_DIR,
            exp_list='train_set.untampered_bulk_primary_cells_with_data'
        )
    output:
        '{env_dir}/data/experiment_lists/{exp_list}/trained_models/{algo_config}/model.pickle'.format(
            env_dir=ENV_DIR,
            exp_list='train_set.untampered_bulk_primary_cells_with_data',
            algo_config=BNC_LOG_CPM_LINEAR_SVM_NORMAL
        )
    run:
        commands = [
            'mkdir -p /scratch/mnbernstein/cell_type_phenotyping_tmp/train_and_save_artifacts',
            'mkdir -p {env_dir}/data/experiment_lists/{exp_list}/trained_models/{algo_config}'.format(
                env_dir=ENV_DIR,
                exp_list='train_set.untampered_bulk_primary_cells_with_data',
                algo_config=BNC_LOG_CPM_LINEAR_SVM_NORMAL
            ),
            'python2.7 {project_dir}/learning/train_and_save.py {project_dir}/learning/train_and_save_config/{env_name}/{algo_config}.json {env_dir} {exp_list} -o {env_dir}/data/experiment_lists/{exp_list}/trained_models/{algo_config} -r /scratch/mnbernstein/cell_type_phenotyping_tmp/train_and_save_artifacts'.format(
                env_dir=ENV_DIR,
                env_name=ENV_NAME,
                exp_list='train_set.untampered_bulk_primary_cells_with_data',
                algo_config=BNC_LOG_CPM_LINEAR_SVM_NORMAL,
                project_dir=PROJECT_DIR
            )
        ]
        for c in commands:
            shell(c)


rule train_toy_bnc_linear_svm_normal_count_cond_prior:
    input:
        '{env_dir}/data/experiment_lists/{exp_list}/experiment_list.json'.format(
            env_dir=ENV_DIR,
            exp_list='toy'
        ),
        '{env_dir}/data/experiment_lists/{exp_list}/labelling.json'.format(
            env_dir=ENV_DIR,
            exp_list='toy'
        )
    output:
        '{env_dir}/data/experiment_lists/{exp_list}/trained_models/{algo_config}/model.pickle'.format(
            env_dir=ENV_DIR,
            exp_list='toy',
            algo_config=BNC_LOG_CPM_LINEAR_SVM_NORMAL
        )
    run:
        commands = [
            'mkdir -p /scratch/mnbernstein/cell_type_phenotyping_tmp/train_and_save_artifacts',
            'mkdir -p {env_dir}/data/experiment_lists/{exp_list}/trained_models/{algo_config}'.format(
                env_dir=ENV_DIR,
                exp_list='toy',
                algo_config=BNC_LOG_CPM_LINEAR_SVM_NORMAL
            ),
            'python2.7 {project_dir}/learning/train_and_save.py {project_dir}/learning/train_and_save_config/{env_name}/{algo_config}.json {env_dir} {exp_list} -o {env_dir}/data/experiment_lists/{exp_list}/trained_models/{algo_config} -r /scratch/mnbernstein/cell_type_phenotyping_tmp/train_and_save_artifacts'.format(
                env_dir=ENV_DIR,
                env_name=ENV_NAME,
                exp_list='toy',
                algo_config=BNC_LOG_CPM_LINEAR_SVM_NORMAL,
                project_dir=PROJECT_DIR
            )
        ]
        for c in commands:
            shell(c)


rule train_bnc_linear_svm_normal_fixed_std_count_cond_prior:
    input:
        '{env_dir}/data/experiment_lists/{exp_list}/experiment_list.json'.format(
            env_dir=ENV_DIR,
            exp_list='train_set.untampered_bulk_primary_cells_with_data'
        ),
        '{env_dir}/data/experiment_lists/{exp_list}/labelling.json'.format(
            env_dir=ENV_DIR,
            exp_list='train_set.untampered_bulk_primary_cells_with_data'
        )
    output:
        '{env_dir}/data/experiment_lists/{exp_list}/trained_models/{algo_config}/model.pickle'.format(
            env_dir=ENV_DIR,
            exp_list='train_set.untampered_bulk_primary_cells_with_data',
            algo_config=BNC_LOG_CPM_LINEAR_SVM_NORMAL_FIXED_STD
        )
    run:
        commands = [
            'mkdir -p /scratch/mnbernstein/cell_type_phenotyping_tmp/train_and_save_artifacts',
            'mkdir -p {env_dir}/data/experiment_lists/{exp_list}/trained_models/{algo_config}'.format(
                env_dir=ENV_DIR,
                exp_list='train_set.untampered_bulk_primary_cells_with_data',
                algo_config=BNC_LOG_CPM_LINEAR_SVM_NORMAL_FIXED_STD
            ),
            'python2.7 {project_dir}/learning/train_and_save.py {project_dir}/learning/train_and_save_config/{env_name}/{algo_config}.json {env_dir} {exp_list} -o {env_dir}/data/experiment_lists/{exp_list}/trained_models/{algo_config} -r /scratch/mnbernstein/cell_type_phenotyping_tmp/train_and_save_artifacts'.format(
                env_dir=ENV_DIR,
                env_name=ENV_NAME,
                exp_list='train_set.untampered_bulk_primary_cells_with_data',
                algo_config=BNC_LOG_CPM_LINEAR_SVM_NORMAL_FIXED_STD,
                project_dir=PROJECT_DIR
            )
        ]
        for c in commands:
            shell(c)


rule train_full_data_bnc_linear_svm_count_cond_prior:
    input:
        '{env_dir}/data/experiment_lists/{exp_list}/experiment_list.json'.format(
            env_dir=ENV_DIR,
            exp_list='untampered_bulk_primary_cells_with_data'
        ),
        '{env_dir}/data/experiment_lists/{exp_list}/labelling.json'.format(
            env_dir=ENV_DIR,
            exp_list='untampered_bulk_primary_cells_with_data'
        )
    output:
        '{env_dir}/data/experiment_lists/{exp_list}/trained_models/{algo_config}/model.pickle'.format(
            env_dir=ENV_DIR,
            exp_list='untampered_bulk_primary_cells_with_data',
            algo_config=BNC_LOG_CPM_LINEAR_SVM_COND
        )
    run:
        commands = [
            'mkdir -p /scratch/mnbernstein/cell_type_phenotyping_tmp/train_and_save_artifacts',
            'mkdir -p {env_dir}/data/experiment_lists/{exp_list}/trained_models/{algo_config}'.format(
                env_dir=ENV_DIR,
                exp_list='untampered_bulk_primary_cells_with_data',
                algo_config=BNC_LOG_CPM_LINEAR_SVM_COND
            ),
            'python2.7 {project_dir}/learning/train_and_save.py {project_dir}/learning/train_and_save_config/{env_name}/{algo_config}.json {env_dir} {exp_list} -o {env_dir}/data/experiment_lists/{exp_list}/trained_models/{algo_config} -r /scratch/mnbernstein/cell_type_phenotyping_tmp/train_and_save_artifacts'.format(
                env_dir=ENV_DIR,
                env_name=ENV_NAME,
                exp_list='untampered_bulk_primary_cells_with_data',
                algo_config=BNC_LOG_CPM_LINEAR_SVM_COND,
                project_dir=PROJECT_DIR
            )
        ]
        for c in commands:
            shell(c)


rule train_tpr_logistic_regression_l2_assert_ambig_neg:
    input:
        '{env_dir}/data/experiment_lists/{exp_list}/experiment_list.json'.format(
            env_dir=ENV_DIR,
            exp_list='train_set.untampered_bulk_primary_cells_with_data'
        ),
        '{env_dir}/data/experiment_lists/{exp_list}/labelling.json'.format(
            env_dir=ENV_DIR,
            exp_list='train_set.untampered_bulk_primary_cells_with_data'
        )
    output:
        '{env_dir}/data/experiment_lists/{exp_list}/trained_models/{algo_config}/model.pickle'.format(
            env_dir=ENV_DIR,
            exp_list='train_set.untampered_bulk_primary_cells_with_data',
            algo_config=TPR_LOGISTIC_L2_LOG_CPM_FULL
        )
    run:
        commands = [
            'mkdir -p /scratch/mnbernstein/cell_type_phenotyping_tmp/train_and_save_artifacts',
            'mkdir -p {env_dir}/data/experiment_lists/{exp_list}/trained_models/{algo_config}'.format(
                env_dir=ENV_DIR,
                exp_list='train_set.untampered_bulk_primary_cells_with_data',
                algo_config=TPR_LOGISTIC_L2_LOG_CPM_FULL
            ),
            'python2.7 {project_dir}/learning/train_and_save.py {project_dir}/learning/train_and_save_config/{env_name}/{algo_config}.json {env_dir} {exp_list} -o {env_dir}/data/experiment_lists/{exp_list}/trained_models/{algo_config} -r /scratch/mnbernstein/cell_type_phenotyping_tmp/train_and_save_artifacts'.format(
                env_dir=ENV_DIR,
                env_name=ENV_NAME,
                exp_list='train_set.untampered_bulk_primary_cells_with_data',
                algo_config=TPR_LOGISTIC_L2_LOG_CPM_FULL,
                project_dir=PROJECT_DIR
            )
        ]
        for c in commands:
            shell(c)

rule train_full_data_tpr_logistic_regression_l2_assert_ambig_neg:
    input:
        '{env_dir}/data/experiment_lists/{exp_list}/experiment_list.json'.format(
            env_dir=ENV_DIR,
            exp_list='untampered_bulk_primary_cells_with_data'
        ),
        '{env_dir}/data/experiment_lists/{exp_list}/labelling.json'.format(
            env_dir=ENV_DIR,
            exp_list='untampered_bulk_primary_cells_with_data'
        )
    output:
        '{env_dir}/data/experiment_lists/{exp_list}/trained_models/{algo_config}/model.pickle'.format(
            env_dir=ENV_DIR,
            exp_list='untampered_bulk_primary_cells_with_data',
            algo_config=TPR_LOGISTIC_L2_LOG_CPM_FULL
        )
    run:
        commands = [
            'mkdir -p /scratch/mnbernstein/cell_type_phenotyping_tmp/train_and_save_artifacts',
            'mkdir -p {env_dir}/data/experiment_lists/{exp_list}/trained_models/{algo_config}'.format(
                env_dir=ENV_DIR,
                exp_list='untampered_bulk_primary_cells_with_data',
                algo_config=TPR_LOGISTIC_L2_LOG_CPM_FULL
            ),
            'python2.7 {project_dir}/learning/train_and_save.py {project_dir}/learning/train_and_save_config/{env_name}/{algo_config}.json {env_dir} {exp_list} -o {env_dir}/data/experiment_lists/{exp_list}/trained_models/{algo_config} -r /scratch/mnbernstein/cell_type_phenotyping_tmp/train_and_save_artifacts'.format(
                env_dir=ENV_DIR,
                env_name=ENV_NAME,
                exp_list='untampered_bulk_primary_cells_with_data',
                algo_config=TPR_LOGISTIC_L2_LOG_CPM_FULL,
                project_dir=PROJECT_DIR
            )
        ]
        for c in commands:
            shell(c)

rule train_tpr_logistic_regression_l2_assert_ambig_neg_downweight_by_group:
    input:
        '{env_dir}/data/experiment_lists/{exp_list}/experiment_list.json'.format(
            env_dir=ENV_DIR,
            exp_list='train_set.untampered_bulk_primary_cells_with_data'
        ),
        '{env_dir}/data/experiment_lists/{exp_list}/labelling.json'.format(
            env_dir=ENV_DIR,
            exp_list='train_set.untampered_bulk_primary_cells_with_data'
        )
    output:
        '{env_dir}/data/experiment_lists/{exp_list}/trained_models/{algo_config}/model.pickle'.format(
            env_dir=ENV_DIR,
            exp_list='train_set.untampered_bulk_primary_cells_with_data',
            algo_config=TPR_LOGISTIC_L2_LOG_CPM_FULL_DOWNWEIGHT
        )
    run:
        commands = [
            'mkdir -p /scratch/mnbernstein/cell_type_phenotyping_tmp/train_and_save_artifacts',
            'mkdir -p {env_dir}/data/experiment_lists/{exp_list}/trained_models/{algo_config}'.format(
                env_dir=ENV_DIR,
                exp_list='train_set.untampered_bulk_primary_cells_with_data',
                algo_config=TPR_LOGISTIC_L2_LOG_CPM_FULL_DOWNWEIGHT
            ),
            'python2.7 {project_dir}/learning/train_and_save.py {project_dir}/learning/train_and_save_config/{env_name}/{algo_config}.json {env_dir} {exp_list} -o {env_dir}/data/experiment_lists/{exp_list}/trained_models/{algo_config} -r /scratch/mnbernstein/cell_type_phenotyping_tmp/train_and_save_artifacts'.format(
                env_dir=ENV_DIR,
                env_name=ENV_NAME,
                exp_list='train_set.untampered_bulk_primary_cells_with_data',
                algo_config=TPR_LOGISTIC_L2_LOG_CPM_FULL_DOWNWEIGHT,
                project_dir=PROJECT_DIR
            )
        ]
        for c in commands:
            shell(c)

rule train_full_data_tpr_logistic_regression_l2_assert_ambig_neg_downweight_by_group:
    input:
        '{env_dir}/data/experiment_lists/{exp_list}/experiment_list.json'.format(
            env_dir=ENV_DIR,
            exp_list='untampered_bulk_primary_cells_with_data'
        ),
        '{env_dir}/data/experiment_lists/{exp_list}/labelling.json'.format(
            env_dir=ENV_DIR,
            exp_list='untampered_bulk_primary_cells_with_data'
        )
    output:
        '{env_dir}/data/experiment_lists/{exp_list}/trained_models/{algo_config}/model.pickle'.format(
            env_dir=ENV_DIR,
            exp_list='untampered_bulk_primary_cells_with_data',
            algo_config=TPR_LOGISTIC_L2_LOG_CPM_FULL_DOWNWEIGHT
        )
    run:
        commands = [
            'mkdir -p /scratch/mnbernstein/cell_type_phenotyping_tmp/train_and_save_artifacts',
            'mkdir -p {env_dir}/data/experiment_lists/{exp_list}/trained_models/{algo_config}'.format(
                env_dir=ENV_DIR,
                exp_list='untampered_bulk_primary_cells_with_data',
                algo_config=TPR_LOGISTIC_L2_LOG_CPM_FULL_DOWNWEIGHT
            ),
            'python2.7 {project_dir}/learning/train_and_save.py {project_dir}/learning/train_and_save_config/{env_name}/{algo_config}.json {env_dir} {exp_list} -o {env_dir}/data/experiment_lists/{exp_list}/trained_models/{algo_config} -r /scratch/mnbernstein/cell_type_phenotyping_tmp/train_and_save_artifacts'.format(
                env_dir=ENV_DIR,
                env_name=ENV_NAME,
                exp_list='untampered_bulk_primary_cells_with_data',
                algo_config=TPR_LOGISTIC_L2_LOG_CPM_FULL_DOWNWEIGHT,
                project_dir=PROJECT_DIR
            )
        ]
        for c in commands:
            shell(c)

rule train_bnc_logistic_regression_l2_assert_ambig_neg_cond_prior:
    input:
        '{env_dir}/data/experiment_lists/{exp_list}/experiment_list.json'.format(
            env_dir=ENV_DIR,
            exp_list='train_set.untampered_bulk_primary_cells_with_data'
        ),
        '{env_dir}/data/experiment_lists/{exp_list}/labelling.json'.format(
            env_dir=ENV_DIR,
            exp_list='train_set.untampered_bulk_primary_cells_with_data'
        )
    output:
        '{env_dir}/data/experiment_lists/{exp_list}/trained_models/{algo_config}/model.pickle'.format(
            env_dir=ENV_DIR,
            exp_list='train_set.untampered_bulk_primary_cells_with_data',
            algo_config=BNC_LOG_CPM_LOGISTIC_L2_COND
        )
    run:
        commands = [
            'mkdir -p /scratch/mnbernstein/cell_type_phenotyping_tmp/train_and_save_artifacts',
            'mkdir -p {env_dir}/data/experiment_lists/{exp_list}/trained_models/{algo_config}'.format(
                env_dir=ENV_DIR,
                exp_list='train_set.untampered_bulk_primary_cells_with_data',
                algo_config=BNC_LOG_CPM_LOGISTIC_L2_COND
            ),
            'python2.7 {project_dir}/learning/train_and_save.py {project_dir}/learning/train_and_save_config/{env_name}/{algo_config}.json {env_dir} {exp_list} -o {env_dir}/data/experiment_lists/{exp_list}/trained_models/{algo_config} -r /scratch/mnbernstein/cell_type_phenotyping_tmp/train_and_save_artifacts'.format(
                env_dir=ENV_DIR,
                env_name=ENV_NAME,
                exp_list='train_set.untampered_bulk_primary_cells_with_data',
                algo_config=BNC_LOG_CPM_LOGISTIC_L2_COND,
                project_dir=PROJECT_DIR
            )
        ]
        for c in commands:
            shell(c)

rule train_bnc_logistic_regression_l2_cond_prior:
    input:
        '{env_dir}/data/experiment_lists/{exp_list}/experiment_list.json'.format(
            env_dir=ENV_DIR,
            exp_list='train_set.untampered_bulk_primary_cells_with_data'
        ),
        '{env_dir}/data/experiment_lists/{exp_list}/labelling.json'.format(
            env_dir=ENV_DIR,
            exp_list='train_set.untampered_bulk_primary_cells_with_data'
        )
    output:
        '{env_dir}/data/experiment_lists/{exp_list}/trained_models/{algo_config}/model.pickle'.format(
            env_dir=ENV_DIR,
            exp_list='train_set.untampered_bulk_primary_cells_with_data',
            algo_config=BNc_LOG_CPM_LOGISTIC_L2_COND_NO_ASSERT_AMBIG_NEG
        )
    run:
        commands = [
            'mkdir -p /scratch/mnbernstein/cell_type_phenotyping_tmp/train_and_save_artifacts',
            'mkdir -p {env_dir}/data/experiment_lists/{exp_list}/trained_models/{algo_config}'.format(
                env_dir=ENV_DIR,
                exp_list='train_set.untampered_bulk_primary_cells_with_data',
                algo_config=BNc_LOG_CPM_LOGISTIC_L2_COND_NO_ASSERT_AMBIG_NEG
            ),
            'python2.7 {project_dir}/learning/train_and_save.py {project_dir}/learning/train_and_save_config/{env_name}/{algo_config}.json {env_dir} {exp_list} -o {env_dir}/data/experiment_lists/{exp_list}/trained_models/{algo_config} -r /scratch/mnbernstein/cell_type_phenotyping_tmp/train_and_save_artifacts'.format(
                env_dir=ENV_DIR,
                env_name=ENV_NAME,
                exp_list='train_set.untampered_bulk_primary_cells_with_data',
                algo_config=BNc_LOG_CPM_LOGISTIC_L2_COND_NO_ASSERT_AMBIG_NEG,
                project_dir=PROJECT_DIR
            )
        ]
        for c in commands:
            shell(c)
    

#rule train_bnc_logistic_regression_l2_assert_ambig_neg:
#    input:
#        '{env_dir}/data/experiment_lists/{exp_list}/experiment_list.json'.format(
#            env_dir=ENV_DIR,
#            exp_list='train_set.untampered_bulk_primary_cells_with_data'
#        ),
#        '{env_dir}/data/experiment_lists/{exp_list}/labelling.json'.format(
#            env_dir=ENV_DIR,
#            exp_list='train_set.untampered_bulk_primary_cells_with_data'
#        )
#    output:
#        '{env_dir}/data/experiment_lists/{exp_list}/trained_models/{algo_config}/model.pickle'.format(
#            env_dir=ENV_DIR,
#            exp_list='train_set.untampered_bulk_primary_cells_with_data',
#            algo_config=BNC_LOG_CPM_LOGISTIC_L2_COND
#        )
#    run:
#        commands = [
#            'mkdir -p /scratch/mnbernstein/cell_type_phenotyping_tmp/train_and_save_artifacts',
#            'mkdir -p {env_dir}/data/experiment_lists/{exp_list}/trained_models/{algo_config}'.format(
#                env_dir=ENV_DIR,
#                exp_list='train_set.untampered_bulk_primary_cells_with_data',
#                algo_config=BNC_LOG_CPM_LOGISTIC_L2_COND
#            ),
#            'python2.7 {project_dir}/learning/train_and_save.py {project_dir}/learning/train_and_save_config/{env_name}/{algo_config}.json {env_dir} {exp_list} -o {env_dir}/data/experiment_lists/{exp_list}/trained_models/{algo_config} -r /scratch/mnbernstein/cell_type_phenotyping_tmp/train_and_save_artifacts'.format(
#                env_dir=ENV_DIR,
#                env_name=ENV_NAME,
#                exp_list='train_set.untampered_bulk_primary_cells_with_data',
#                algo_config=BNC_LOG_CPM_LOGISTIC_L2_COND,
#                project_dir=PROJECT_DIR
#            )
#        ]
#        for c in commands:
#            shell(c)
#        )

rule train_bnc_logistic_regression_l2_assert_ambig_neg_downweight_by_group:
    input:
        '{env_dir}/data/experiment_lists/{exp_list}/experiment_list.json'.format(
            env_dir=ENV_DIR,
            exp_list='train_set.untampered_bulk_primary_cells_with_data'
        ),
        '{env_dir}/data/experiment_lists/{exp_list}/labelling.json'.format(
            env_dir=ENV_DIR,
            exp_list='train_set.untampered_bulk_primary_cells_with_data'
        )
    output:
        '{env_dir}/data/experiment_lists/{exp_list}/trained_models/{algo_config}/model.pickle'.format(
            env_dir=ENV_DIR,
            exp_list='train_set.untampered_bulk_primary_cells_with_data',
            algo_config=BNC_LOG_CPM_LOGISTIC_L2_DOWNWEIGHT
        )
    run:
        commands = [
            'mkdir -p /scratch/mnbernstein/cell_type_phenotyping_tmp/train_and_save_artifacts',
            'mkdir -p {env_dir}/data/experiment_lists/{exp_list}/trained_models/{algo_config}'.format(
                env_dir=ENV_DIR,
                exp_list='train_set.untampered_bulk_primary_cells_with_data',
                algo_config=BNC_LOG_CPM_LOGISTIC_L2_DOWNWEIGHT
            ),
            'python2.7 {project_dir}/learning/train_and_save.py {project_dir}/learning/train_and_save_config/{env_name}/{algo_config}.json {env_dir} {exp_list} -o {env_dir}/data/experiment_lists/{exp_list}/trained_models/{algo_config} -r /scratch/mnbernstein/cell_type_phenotyping_tmp/train_and_save_artifacts'.format(
                env_dir=ENV_DIR,
                env_name=ENV_NAME,
                exp_list='train_set.untampered_bulk_primary_cells_with_data',
                algo_config=BNC_LOG_CPM_LOGISTIC_L2_DOWNWEIGHT,
                project_dir=PROJECT_DIR
            )
        ]
        for c in commands:
            shell(c)


rule train_bnc_logistic_regression_l2_assert_ambig_neg_conditional_downweight_by_group:
    input:
        '{env_dir}/data/experiment_lists/{exp_list}/experiment_list.json'.format(
            env_dir=ENV_DIR,
            exp_list='train_set.untampered_bulk_primary_cells_with_data'
        ),
        '{env_dir}/data/experiment_lists/{exp_list}/labelling.json'.format(
            env_dir=ENV_DIR,
            exp_list='train_set.untampered_bulk_primary_cells_with_data'
        )
    output:
        '{env_dir}/data/experiment_lists/{exp_list}/trained_models/{algo_config}/model.pickle'.format(
            env_dir=ENV_DIR,
            exp_list='train_set.untampered_bulk_primary_cells_with_data',
            algo_config=BNC_LOG_CPM_LOGISTIC_L2_COND_DOWNDWIEGHT
        )
    run:
        commands = [
            'mkdir -p /scratch/mnbernstein/cell_type_phenotyping_tmp/train_and_save_artifacts',
            'mkdir -p {env_dir}/data/experiment_lists/{exp_list}/trained_models/{algo_config}'.format(
                env_dir=ENV_DIR,
                exp_list='train_set.untampered_bulk_primary_cells_with_data',
                algo_config=BNC_LOG_CPM_LOGISTIC_L2_COND_DOWNDWIEGHT
            ),
            'python2.7 {project_dir}/learning/train_and_save.py {project_dir}/learning/train_and_save_config/{env_name}/{algo_config}.json {env_dir} {exp_list} -o {env_dir}/data/experiment_lists/{exp_list}/trained_models/{algo_config} -r /scratch/mnbernstein/cell_type_phenotyping_tmp/train_and_save_artifacts'.format(
                env_dir=ENV_DIR,
                env_name=ENV_NAME,
                exp_list='train_set.untampered_bulk_primary_cells_with_data',
                algo_config=BNC_LOG_CPM_LOGISTIC_L2_COND_DOWNDWIEGHT,
                project_dir=PROJECT_DIR
            )
        ]
        for c in commands:
            shell(c)

#####################################################################
#   Apply the saved models to their respective test sests
#####################################################################

rule apply_trained_model_naive_bayes_cond_assert_ambig_neg:
    input:
        test_exp_list = '{env_dir}/data/experiment_lists/{exp_list}/experiment_list.json'.format(
            env_dir=ENV_DIR,
            exp_list='test_set.untampered_bulk_primary_cells_with_data'
        ),
        labelling_f = '{env_dir}/data/experiment_lists/{exp_list}/labelling.json'.format(
            env_dir=ENV_DIR,
            exp_list='test_set.untampered_bulk_primary_cells_with_data'
        ),
        model_f = '{env_dir}/data/experiment_lists/{train_exp_list}/trained_models/{algo_config}/model.pickle'.format(
            env_dir=ENV_DIR,
            train_exp_list='train_set.untampered_bulk_primary_cells_with_data',
            algo_config=NAIVE_BAYES_SVM_COND_ASSERT_AMBIG_NEG
        )
    output:
        '{env_dir}/data/experiment_lists/{exp_list}/trained_on_another_experiment_list/{train_exp_list}/{algo_config}/prediction_results.json'.format(
            env_dir=ENV_DIR,
            exp_list='test_set.untampered_bulk_primary_cells_with_data',
            train_exp_list='train_set.untampered_bulk_primary_cells_with_data',
            algo_config=NAIVE_BAYES_SVM_COND_ASSERT_AMBIG_NEG
        )
    run:
        commands = [
            'mkdir -p {env_dir}/data/experiment_lists/{exp_list}/trained_on_another_experiment_list/{train_exp_list}/{algo_config}'.format(
                env_dir=ENV_DIR,
                exp_list='test_set.untampered_bulk_primary_cells_with_data',
                train_exp_list='train_set.untampered_bulk_primary_cells_with_data',
                algo_config=NAIVE_BAYES_SVM_COND_ASSERT_AMBIG_NEG
            ),
            'python2.7 {project_dir}/learning/apply_saved_model.py {{input.model_f}} {env_dir} {exp_list} -o {env_dir}/data/experiment_lists/{exp_list}/trained_on_another_experiment_list/{train_exp_list}/{algo_config}'.format(
                env_dir=ENV_DIR,
                project_dir=PROJECT_DIR,
                exp_list='test_set.untampered_bulk_primary_cells_with_data',
                train_exp_list='train_set.untampered_bulk_primary_cells_with_data',
                algo_config=NAIVE_BAYES_SVM_COND_ASSERT_AMBIG_NEG
            )
        ]
        for c in commands:
            shell(c)

rule apply_trained_model_naive_bayes_counts_assert_ambig_neg:
    input:
        test_exp_list = '{env_dir}/data/experiment_lists/{exp_list}/experiment_list.json'.format(
            env_dir=ENV_DIR,
            exp_list='test_set.untampered_bulk_primary_cells_with_data'
        ),
        labelling_f = '{env_dir}/data/experiment_lists/{exp_list}/labelling.json'.format(
            env_dir=ENV_DIR,
            exp_list='test_set.untampered_bulk_primary_cells_with_data'
        ),
        model_f = '{env_dir}/data/experiment_lists/{train_exp_list}/trained_models/{algo_config}/model.pickle'.format(
            env_dir=ENV_DIR,
            train_exp_list='train_set.untampered_bulk_primary_cells_with_data',
            algo_config=NAIVE_BAYES_SVM_COUNTS_ASSERT_AMBIG_NEG
        )
    output:
        '{env_dir}/data/experiment_lists/{exp_list}/trained_on_another_experiment_list/{train_exp_list}/{algo_config}/prediction_results.json'.format(
            env_dir=ENV_DIR,
            exp_list='test_set.untampered_bulk_primary_cells_with_data',
            train_exp_list='train_set.untampered_bulk_primary_cells_with_data',
            algo_config=NAIVE_BAYES_SVM_COUNTS_ASSERT_AMBIG_NEG
        )
    run:
        commands = [
            'mkdir -p {env_dir}/data/experiment_lists/{exp_list}/trained_on_another_experiment_list/{train_exp_list}/{algo_config}'.format(
                env_dir=ENV_DIR,
                exp_list='test_set.untampered_bulk_primary_cells_with_data',
                train_exp_list='train_set.untampered_bulk_primary_cells_with_data',
                algo_config=NAIVE_BAYES_SVM_COUNTS_ASSERT_AMBIG_NEG
            ),
            'python2.7 {project_dir}/learning/apply_saved_model.py {{input.model_f}} {env_dir} {exp_list} -o {env_dir}/data/experiment_lists/{exp_list}/trained_on_another_experiment_list/{train_exp_list}/{algo_config}'.format(
                env_dir=ENV_DIR,
                project_dir=PROJECT_DIR,
                exp_list='test_set.untampered_bulk_primary_cells_with_data',
                train_exp_list='train_set.untampered_bulk_primary_cells_with_data',
                algo_config=NAIVE_BAYES_SVM_COUNTS_ASSERT_AMBIG_NEG
            )
        ]
        for c in commands:
            shell(c)

rule apply_trained_model_per_label_logistic_regression_l2_assert_ambig_neg:
    input:
        test_exp_list = '{env_dir}/data/experiment_lists/{exp_list}/experiment_list.json'.format(
            env_dir=ENV_DIR,
            exp_list='test_set.untampered_bulk_primary_cells_with_data'
        ),
        labelling_f = '{env_dir}/data/experiment_lists/{exp_list}/labelling.json'.format(
            env_dir=ENV_DIR,
            exp_list='test_set.untampered_bulk_primary_cells_with_data'
        ),
        model_f = '{env_dir}/data/experiment_lists/{train_exp_list}/trained_models/{algo_config}/model.pickle'.format(
            env_dir=ENV_DIR,
            train_exp_list='train_set.untampered_bulk_primary_cells_with_data',
            algo_config=PER_LABEL_LOG_CPM_LOGISTIC_L2_FULL
        )
    output:
        '{env_dir}/data/experiment_lists/{exp_list}/trained_on_another_experiment_list/{train_exp_list}/{algo_config}/prediction_results.json'.format(
            env_dir=ENV_DIR,
            exp_list='test_set.untampered_bulk_primary_cells_with_data',
            train_exp_list='train_set.untampered_bulk_primary_cells_with_data',
            algo_config=PER_LABEL_LOG_CPM_LOGISTIC_L2_FULL
        )
    run:
        commands = [
            'mkdir -p {env_dir}/data/experiment_lists/{exp_list}/trained_on_another_experiment_list/{train_exp_list}/{algo_config}'.format(
                env_dir=ENV_DIR,
                exp_list='test_set.untampered_bulk_primary_cells_with_data',
                train_exp_list='train_set.untampered_bulk_primary_cells_with_data',
                algo_config=PER_LABEL_LOG_CPM_LOGISTIC_L2_FULL
            ),
            'python2.7 {project_dir}/learning/apply_saved_model.py {{input.model_f}} {env_dir} {exp_list} -o {env_dir}/data/experiment_lists/{exp_list}/trained_on_another_experiment_list/{train_exp_list}/{algo_config}'.format(
                env_dir=ENV_DIR,
                project_dir=PROJECT_DIR,
                exp_list='test_set.untampered_bulk_primary_cells_with_data',
                train_exp_list='train_set.untampered_bulk_primary_cells_with_data',
                algo_config=PER_LABEL_LOG_CPM_LOGISTIC_L2_FULL
            )
        ]
        for c in commands:
            shell(c)

rule apply_trained_model_per_label_logistic_regression_l2_assert_ambig_neg_downweight:
    input:
        test_exp_list = '{env_dir}/data/experiment_lists/{exp_list}/experiment_list.json'.format(
            env_dir=ENV_DIR,
            exp_list='test_set.untampered_bulk_primary_cells_with_data'
        ),
        labelling_f = '{env_dir}/data/experiment_lists/{exp_list}/labelling.json'.format(
            env_dir=ENV_DIR,
            exp_list='test_set.untampered_bulk_primary_cells_with_data'
        ),
        model_f = '{env_dir}/data/experiment_lists/{train_exp_list}/trained_models/{algo_config}/model.pickle'.format(
            env_dir=ENV_DIR,
            train_exp_list='train_set.untampered_bulk_primary_cells_with_data',
            algo_config=PER_LABEL_LOG_CPM_LOGISTIC_L2_FULL_DOWNWEIGHT 
        )
    output:
        '{env_dir}/data/experiment_lists/{exp_list}/trained_on_another_experiment_list/{train_exp_list}/{algo_config}/prediction_results.json'.format(
            env_dir=ENV_DIR,
            exp_list='test_set.untampered_bulk_primary_cells_with_data',
            train_exp_list='train_set.untampered_bulk_primary_cells_with_data',
            algo_config=PER_LABEL_LOG_CPM_LOGISTIC_L2_FULL_DOWNWEIGHT 
        )
    run:
        commands = [
            'mkdir -p {env_dir}/data/experiment_lists/{exp_list}/trained_on_another_experiment_list/{train_exp_list}/{algo_config}'.format(
                env_dir=ENV_DIR,
                exp_list='test_set.untampered_bulk_primary_cells_with_data',
                train_exp_list='train_set.untampered_bulk_primary_cells_with_data',
                algo_config=PER_LABEL_LOG_CPM_LOGISTIC_L2_FULL_DOWNWEIGHT 
            ),
            'python2.7 {project_dir}/learning/apply_saved_model.py {{input.model_f}} {env_dir} {exp_list} -o {env_dir}/data/experiment_lists/{exp_list}/trained_on_another_experiment_list/{train_exp_list}/{algo_config}'.format(
                env_dir=ENV_DIR,
                project_dir=PROJECT_DIR,
                exp_list='test_set.untampered_bulk_primary_cells_with_data',
                train_exp_list='train_set.untampered_bulk_primary_cells_with_data',
                algo_config=PER_LABEL_LOG_CPM_LOGISTIC_L2_FULL_DOWNWEIGHT 
            )
        ]
        for c in commands:
            shell(c)


rule toy_apply_trained_model_cdc_logistic_regression_l2_assert_ambig_neg:
    input:
        test_exp_list = '{env_dir}/data/experiment_lists/{exp_list}/experiment_list.json'.format(
            env_dir=ENV_DIR,
            exp_list='toy'
        ),
        labelling_f = '{env_dir}/data/experiment_lists/{exp_list}/labelling.json'.format(
            env_dir=ENV_DIR,
            exp_list='toy'
        ),
        model_f = '{env_dir}/data/experiment_lists/{train_exp_list}/trained_models/{algo_config}/model.pickle'.format(
            env_dir=ENV_DIR,
            train_exp_list='toy',
            algo_config=CDC_LOG_CPM_LOGISTIC_L2_ASSERT_AMBIG_NEG_FULL
        )
    output:
        '{env_dir}/data/experiment_lists/{exp_list}/trained_on_another_experiment_list/{train_exp_list}/{algo_config}/prediction_results.json'.format(
            env_dir=ENV_DIR,
            exp_list='toy',
            train_exp_list='toy',
            algo_config=CDC_LOG_CPM_LOGISTIC_L2_ASSERT_AMBIG_NEG_FULL
        )
    run:
        commands = [
            'mkdir -p {env_dir}/data/experiment_lists/{exp_list}/trained_on_another_experiment_list/{train_exp_list}/{algo_config}'.format(
                env_dir=ENV_DIR,
                exp_list='toy',
                train_exp_list='toy',
                algo_config=CDC_LOG_CPM_LOGISTIC_L2_ASSERT_AMBIG_NEG_FULL
            ),
            'python2.7 {project_dir}/learning/apply_saved_model.py {{input.model_f}} {env_dir} {exp_list} -o {env_dir}/data/experiment_lists/{exp_list}/trained_on_another_experiment_list/{train_exp_list}/{algo_config}'.format(
                env_dir=ENV_DIR,
                project_dir=PROJECT_DIR,
                exp_list='toy',
                train_exp_list='toy',
                algo_config=CDC_LOG_CPM_LOGISTIC_L2_ASSERT_AMBIG_NEG_FULL
            )
        ]
        for c in commands:
            shell(c)

rule apply_trained_model_cdc_logistic_regression_l2_assert_ambig_neg:
    input:
        test_exp_list = '{env_dir}/data/experiment_lists/{exp_list}/experiment_list.json'.format(
            env_dir=ENV_DIR,
            exp_list='test_set.untampered_bulk_primary_cells_with_data'
        ),
        labelling_f = '{env_dir}/data/experiment_lists/{exp_list}/labelling.json'.format(
            env_dir=ENV_DIR,
            exp_list='test_set.untampered_bulk_primary_cells_with_data'
        ),
        model_f = '{env_dir}/data/experiment_lists/{train_exp_list}/trained_models/{algo_config}/model.pickle'.format(
            env_dir=ENV_DIR,
            train_exp_list='train_set.untampered_bulk_primary_cells_with_data',
            algo_config=CDC_LOG_CPM_LOGISTIC_L2_ASSERT_AMBIG_NEG_FULL
        )
    output:
        '{env_dir}/data/experiment_lists/{exp_list}/trained_on_another_experiment_list/{train_exp_list}/{algo_config}/prediction_results.json'.format(
            env_dir=ENV_DIR,
            exp_list='test_set.untampered_bulk_primary_cells_with_data',
            train_exp_list='train_set.untampered_bulk_primary_cells_with_data',
            algo_config=CDC_LOG_CPM_LOGISTIC_L2_ASSERT_AMBIG_NEG_FULL
        )
    run:
        commands = [
            'mkdir -p {env_dir}/data/experiment_lists/{exp_list}/trained_on_another_experiment_list/{train_exp_list}/{algo_config}'.format(
                env_dir=ENV_DIR,
                exp_list='test_set.untampered_bulk_primary_cells_with_data',
                train_exp_list='train_set.untampered_bulk_primary_cells_with_data',
                algo_config=CDC_LOG_CPM_LOGISTIC_L2_ASSERT_AMBIG_NEG_FULL
            ),
            'python2.7 {project_dir}/learning/apply_saved_model.py {{input.model_f}} {env_dir} {exp_list} -o {env_dir}/data/experiment_lists/{exp_list}/trained_on_another_experiment_list/{train_exp_list}/{algo_config}'.format(
                env_dir=ENV_DIR,
                project_dir=PROJECT_DIR,
                exp_list='test_set.untampered_bulk_primary_cells_with_data',
                train_exp_list='train_set.untampered_bulk_primary_cells_with_data',
                algo_config=CDC_LOG_CPM_LOGISTIC_L2_ASSERT_AMBIG_NEG_FULL
            )
        ]
        for c in commands:
            shell(c)

rule apply_trained_model_to_single_cell_cdc_logistic_regression_l2_assert_ambig_neg:
    input:
        test_exp_list = '{env_dir}/data/experiment_lists/{exp_list}/experiment_list.json'.format(
            env_dir=ENV_DIR,
            exp_list='untampered_single_cell_primary_cells_with_data'
        ),
        labelling_f = '{env_dir}/data/experiment_lists/{exp_list}/labelling.json'.format(
            env_dir=ENV_DIR,
            exp_list='untampered_single_cell_primary_cells_with_data'
        ),
        model_f = '{env_dir}/data/experiment_lists/{train_exp_list}/trained_models/{algo_config}/model.pickle'.format(
            env_dir=ENV_DIR,
            train_exp_list='untampered_bulk_primary_cells_with_data',
            algo_config=CDC_LOG_CPM_LOGISTIC_L2_ASSERT_AMBIG_NEG_FULL
        )
    output:
        '{env_dir}/data/experiment_lists/{exp_list}/trained_on_another_experiment_list/{train_exp_list}/{algo_config}/prediction_results.json'.format(
            env_dir=ENV_DIR,
            exp_list='untampered_single_cell_primary_cells_with_data',
            train_exp_list='untampered_bulk_primary_cells_with_data',
            algo_config=CDC_LOG_CPM_LOGISTIC_L2_ASSERT_AMBIG_NEG_FULL
        )
    run:
        commands = [
            'mkdir -p {env_dir}/data/experiment_lists/{exp_list}/trained_on_another_experiment_list/{train_exp_list}/{algo_config}'.format(
                env_dir=ENV_DIR,
                exp_list='untampered_single_cell_primary_cells_with_data',
                train_exp_list='untampered_bulk_primary_cells_with_data',
                algo_config=CDC_LOG_CPM_LOGISTIC_L2_ASSERT_AMBIG_NEG_FULL
            ),
            'python2.7 {project_dir}/learning/apply_saved_model.py {{input.model_f}} {env_dir} {exp_list} -o {env_dir}/data/experiment_lists/{exp_list}/trained_on_another_experiment_list/{train_exp_list}/{algo_config}'.format(
                env_dir=ENV_DIR,
                project_dir=PROJECT_DIR,
                exp_list='untampered_single_cell_primary_cells_with_data',
                train_exp_list='untampered_bulk_primary_cells_with_data',
                algo_config=CDC_LOG_CPM_LOGISTIC_L2_ASSERT_AMBIG_NEG_FULL
            )
        ]
        for c in commands:
            shell(c)

rule apply_trained_model_cdc_logistic_regression_l2_assert_ambig_neg_downweight_by_group:
    input:
        test_exp_list = '{env_dir}/data/experiment_lists/{exp_list}/experiment_list.json'.format(
            env_dir=ENV_DIR,
            exp_list='test_set.untampered_bulk_primary_cells_with_data'
        ),
        labelling_f = '{env_dir}/data/experiment_lists/{exp_list}/labelling.json'.format(
            env_dir=ENV_DIR,
            exp_list='test_set.untampered_bulk_primary_cells_with_data'
        ),
        model_f = '{env_dir}/data/experiment_lists/{train_exp_list}/trained_models/{algo_config}/model.pickle'.format(
            env_dir=ENV_DIR,
            train_exp_list='train_set.untampered_bulk_primary_cells_with_data',
            algo_config=CDC_LOG_CPM_LOGISTIC_L2_ASSERT_AMBIG_NEG_FULL_DOWNWEIGHT
        )
    output:
        '{env_dir}/data/experiment_lists/{exp_list}/trained_on_another_experiment_list/{train_exp_list}/{algo_config}/prediction_results.json'.format(
            env_dir=ENV_DIR,
            exp_list='test_set.untampered_bulk_primary_cells_with_data',
            train_exp_list='train_set.untampered_bulk_primary_cells_with_data',
            algo_config=CDC_LOG_CPM_LOGISTIC_L2_ASSERT_AMBIG_NEG_FULL_DOWNWEIGHT
        )
    run:
        commands = [
            'mkdir -p {env_dir}/data/experiment_lists/{exp_list}/trained_on_another_experiment_list/{train_exp_list}/{algo_config}'.format(
                env_dir=ENV_DIR,
                exp_list='test_set.untampered_bulk_primary_cells_with_data',
                train_exp_list='train_set.untampered_bulk_primary_cells_with_data',
                algo_config=CDC_LOG_CPM_LOGISTIC_L2_ASSERT_AMBIG_NEG_FULL_DOWNWEIGHT
            ),
            'python2.7 {project_dir}/learning/apply_saved_model.py {{input.model_f}} {env_dir} {exp_list} -o {env_dir}/data/experiment_lists/{exp_list}/trained_on_another_experiment_list/{train_exp_list}/{algo_config}'.format(
                env_dir=ENV_DIR,
                project_dir=PROJECT_DIR,
                exp_list='test_set.untampered_bulk_primary_cells_with_data',
                train_exp_list='train_set.untampered_bulk_primary_cells_with_data',
                algo_config=CDC_LOG_CPM_LOGISTIC_L2_ASSERT_AMBIG_NEG_FULL_DOWNWEIGHT
            )
        ]
        for c in commands:
            shell(c)

rule apply_trained_model_to_single_cell_cdc_logistic_regression_l2_assert_ambig_neg_downweight_by_group:
    input:
        test_exp_list = '{env_dir}/data/experiment_lists/{exp_list}/experiment_list.json'.format(
            env_dir=ENV_DIR,
            exp_list='untampered_single_cell_primary_cells_with_data'
        ),
        labelling_f = '{env_dir}/data/experiment_lists/{exp_list}/labelling.json'.format(
            env_dir=ENV_DIR,
            exp_list='untampered_single_cell_primary_cells_with_data'
        ),
        model_f = '{env_dir}/data/experiment_lists/{train_exp_list}/trained_models/{algo_config}/model.pickle'.format(
            env_dir=ENV_DIR,
            train_exp_list='untampered_bulk_primary_cells_with_data',
            algo_config=CDC_LOG_CPM_LOGISTIC_L2_ASSERT_AMBIG_NEG_FULL_DOWNWEIGHT
        )
    output:
        '{env_dir}/data/experiment_lists/{exp_list}/trained_on_another_experiment_list/{train_exp_list}/{algo_config}/prediction_results.json'.format(
            env_dir=ENV_DIR,
            exp_list='untampered_single_cell_primary_cells_with_data',
            train_exp_list='untampered_bulk_primary_cells_with_data',
            algo_config=CDC_LOG_CPM_LOGISTIC_L2_ASSERT_AMBIG_NEG_FULL_DOWNWEIGHT
        )
    run:
        commands = [
            'mkdir -p {env_dir}/data/experiment_lists/{exp_list}/trained_on_another_experiment_list/{train_exp_list}/{algo_config}'.format(
                env_dir=ENV_DIR,
                exp_list='untampered_single_cell_primary_cells_with_data',
                train_exp_list='untampered_bulk_primary_cells_with_data',
                algo_config=CDC_LOG_CPM_LOGISTIC_L2_ASSERT_AMBIG_NEG_FULL_DOWNWEIGHT
            ),
            'python2.7 {project_dir}/learning/apply_saved_model.py {{input.model_f}} {env_dir} {exp_list} -o {env_dir}/data/experiment_lists/{exp_list}/trained_on_another_experiment_list/{train_exp_list}/{algo_config}'.format(
                env_dir=ENV_DIR,
                project_dir=PROJECT_DIR,
                exp_list='untampered_single_cell_primary_cells_with_data',
                train_exp_list='untampered_bulk_primary_cells_with_data',
                algo_config=CDC_LOG_CPM_LOGISTIC_L2_ASSERT_AMBIG_NEG_FULL_DOWNWEIGHT
            )
        ]
        for c in commands:
            shell(c)

rule apply_trained_model_to_bulk_restricted_single_cell_cdc_logistic_regression_l2_assert_ambig_neg_downweight_by_group:
    input:
        test_exp_list = '{env_dir}/data/experiment_lists/{exp_list}/experiment_list.json'.format(
            env_dir=ENV_DIR,
            exp_list='untampered_single_cell_primary_cells_with_data_cell_types_in_bulk'
        ),
        labelling_f = '{env_dir}/data/experiment_lists/{exp_list}/labelling.json'.format(
            env_dir=ENV_DIR,
            exp_list='untampered_single_cell_primary_cells_with_data_cell_types_in_bulk'
        ),
        model_f = '{env_dir}/data/experiment_lists/{train_exp_list}/trained_models/{algo_config}/model.pickle'.format(
            env_dir=ENV_DIR,
            train_exp_list='untampered_bulk_primary_cells_with_data',
            algo_config=CDC_LOG_CPM_LOGISTIC_L2_ASSERT_AMBIG_NEG_FULL_DOWNWEIGHT
        )
    output:
        '{env_dir}/data/experiment_lists/{exp_list}/trained_on_another_experiment_list/{train_exp_list}/{algo_config}/prediction_results.json'.format(
            env_dir=ENV_DIR,
            exp_list='untampered_single_cell_primary_cells_with_data_cell_types_in_bulk',
            train_exp_list='untampered_bulk_primary_cells_with_data',
            algo_config=CDC_LOG_CPM_LOGISTIC_L2_ASSERT_AMBIG_NEG_FULL_DOWNWEIGHT
        )
    run:
        commands = [
            'mkdir -p {env_dir}/data/experiment_lists/{exp_list}/trained_on_another_experiment_list/{train_exp_list}/{algo_config}'.format(
                env_dir=ENV_DIR,
                exp_list='untampered_single_cell_primary_cells_with_data_cell_types_in_bulk',
                train_exp_list='untampered_bulk_primary_cells_with_data',
                algo_config=CDC_LOG_CPM_LOGISTIC_L2_ASSERT_AMBIG_NEG_FULL_DOWNWEIGHT
            ),
            'python2.7 {project_dir}/learning/apply_saved_model.py {{input.model_f}} {env_dir} {exp_list} -o {env_dir}/data/experiment_lists/{exp_list}/trained_on_another_experiment_list/{train_exp_list}/{algo_config}'.format(
                env_dir=ENV_DIR,
                project_dir=PROJECT_DIR,
                exp_list='untampered_single_cell_primary_cells_with_data_cell_types_in_bulk',
                train_exp_list='untampered_bulk_primary_cells_with_data',
                algo_config=CDC_LOG_CPM_LOGISTIC_L2_ASSERT_AMBIG_NEG_FULL_DOWNWEIGHT
            )
        ]
        for c in commands:
            shell(c)

rule apply_trained_model_isotonic_logistic_regression_l2_assert_ambig_neg:
    input:
        test_exp_list = '{env_dir}/data/experiment_lists/{exp_list}/experiment_list.json'.format(
            env_dir=ENV_DIR,
            exp_list='test_set.untampered_bulk_primary_cells_with_data'
        ),
        labelling_f = '{env_dir}/data/experiment_lists/{exp_list}/labelling.json'.format(
            env_dir=ENV_DIR,
            exp_list='test_set.untampered_bulk_primary_cells_with_data'
        ),
        model_f = '{env_dir}/data/experiment_lists/{train_exp_list}/trained_models/{algo_config}/model.pickle'.format(
            env_dir=ENV_DIR,
            train_exp_list='train_set.untampered_bulk_primary_cells_with_data',
            algo_config=ISOTONIC_LOGISTIC_L2_LOG_CPM_ASSERT_AMBIG_NEG_FULL
        )
    output:
        '{env_dir}/data/experiment_lists/{exp_list}/trained_on_another_experiment_list/{train_exp_list}/{algo_config}/prediction_results.json'.format(
            env_dir=ENV_DIR,
            exp_list='test_set.untampered_bulk_primary_cells_with_data',
            train_exp_list='train_set.untampered_bulk_primary_cells_with_data',
            algo_config=ISOTONIC_LOGISTIC_L2_LOG_CPM_ASSERT_AMBIG_NEG_FULL
        )
    run:
        commands = [
            'mkdir -p {env_dir}/data/experiment_lists/{exp_list}/trained_on_another_experiment_list/{train_exp_list}/{algo_config}'.format(
                env_dir=ENV_DIR,
                exp_list='test_set.untampered_bulk_primary_cells_with_data',
                train_exp_list='train_set.untampered_bulk_primary_cells_with_data',
                algo_config=ISOTONIC_LOGISTIC_L2_LOG_CPM_ASSERT_AMBIG_NEG_FULL
            ),
            'python2.7 {project_dir}/learning/apply_saved_model.py {{input.model_f}} {env_dir} {exp_list} -o {env_dir}/data/experiment_lists/{exp_list}/trained_on_another_experiment_list/{train_exp_list}/{algo_config}'.format(
                env_dir=ENV_DIR,
                project_dir=PROJECT_DIR,
                exp_list='test_set.untampered_bulk_primary_cells_with_data',
                train_exp_list='train_set.untampered_bulk_primary_cells_with_data',
                algo_config=ISOTONIC_LOGISTIC_L2_LOG_CPM_ASSERT_AMBIG_NEG_FULL
            )
        ]
        for c in commands:
            shell(c)



rule apply_trained_model_to_single_cell_isotonic_logistic_regression_l2_assert_ambig_neg:
    input:
        test_exp_list = '{env_dir}/data/experiment_lists/{exp_list}/experiment_list.json'.format(
            env_dir=ENV_DIR,
            exp_list='untampered_single_cell_primary_cells_with_data'
        ),
        labelling_f = '{env_dir}/data/experiment_lists/{exp_list}/labelling.json'.format(
            env_dir=ENV_DIR,
            exp_list='untampered_single_cell_primary_cells_with_data'
        ),
        model_f = '{env_dir}/data/experiment_lists/{train_exp_list}/trained_models/{algo_config}/model.pickle'.format(
            env_dir=ENV_DIR,
            train_exp_list='untampered_bulk_primary_cells_with_data',
            algo_config=ISOTONIC_LOGISTIC_L2_LOG_CPM_ASSERT_AMBIG_NEG_FULL
        )
    output:
        '{env_dir}/data/experiment_lists/{exp_list}/trained_on_another_experiment_list/{train_exp_list}/{algo_config}/prediction_results.json'.format(
            env_dir=ENV_DIR,
            exp_list='untampered_single_cell_primary_cells_with_data',
            train_exp_list='untampered_bulk_primary_cells_with_data',
            algo_config=ISOTONIC_LOGISTIC_L2_LOG_CPM_ASSERT_AMBIG_NEG_FULL
        )
    run:
        commands = [
            'mkdir -p {env_dir}/data/experiment_lists/{exp_list}/trained_on_another_experiment_list/{train_exp_list}/{algo_config}'.format(
                env_dir=ENV_DIR,
                exp_list='untampered_single_cell_primary_cells_with_data',
                train_exp_list='untampered_bulk_primary_cells_with_data',
                algo_config=ISOTONIC_LOGISTIC_L2_LOG_CPM_ASSERT_AMBIG_NEG_FULL
            ),
            'python2.7 {project_dir}/learning/apply_saved_model.py {{input.model_f}} {env_dir} {exp_list} -o {env_dir}/data/experiment_lists/{exp_list}/trained_on_another_experiment_list/{train_exp_list}/{algo_config}'.format(
                env_dir=ENV_DIR,
                project_dir=PROJECT_DIR,
                exp_list='untampered_single_cell_primary_cells_with_data',
                train_exp_list='untampered_bulk_primary_cells_with_data',
                algo_config=ISOTONIC_LOGISTIC_L2_LOG_CPM_ASSERT_AMBIG_NEG_FULL
            )
        ]
        for c in commands:
            shell(c)


rule apply_trained_model_isotonic_logistic_regression_l2_assert_ambig_neg_downweight_by_group:
    input:
        test_exp_list = '{env_dir}/data/experiment_lists/{exp_list}/experiment_list.json'.format(
            env_dir=ENV_DIR,
            exp_list='test_set.untampered_bulk_primary_cells_with_data'
        ),
        labelling_f = '{env_dir}/data/experiment_lists/{exp_list}/labelling.json'.format(
            env_dir=ENV_DIR,
            exp_list='test_set.untampered_bulk_primary_cells_with_data'
        ),
        model_f = '{env_dir}/data/experiment_lists/{train_exp_list}/trained_models/{algo_config}/model.pickle'.format(
            env_dir=ENV_DIR,
            train_exp_list='train_set.untampered_bulk_primary_cells_with_data',
            algo_config=ISOTONIC_LOGISTIC_L2_LOG_CPM_ASSERT_AMBIG_NEG_FULL_DOWNWEIGHT
        )
    output:
        '{env_dir}/data/experiment_lists/{exp_list}/trained_on_another_experiment_list/{train_exp_list}/{algo_config}/prediction_results.json'.format(
            env_dir=ENV_DIR,
            exp_list='test_set.untampered_bulk_primary_cells_with_data',
            train_exp_list='train_set.untampered_bulk_primary_cells_with_data',
            algo_config=ISOTONIC_LOGISTIC_L2_LOG_CPM_ASSERT_AMBIG_NEG_FULL_DOWNWEIGHT
        )
    run:
        commands = [
            'mkdir -p {env_dir}/data/experiment_lists/{exp_list}/trained_on_another_experiment_list/{train_exp_list}/{algo_config}'.format(
                env_dir=ENV_DIR,
                exp_list='test_set.untampered_bulk_primary_cells_with_data',
                train_exp_list='train_set.untampered_bulk_primary_cells_with_data',
                algo_config=ISOTONIC_LOGISTIC_L2_LOG_CPM_ASSERT_AMBIG_NEG_FULL_DOWNWEIGHT
            ),
            'python2.7 {project_dir}/learning/apply_saved_model.py {{input.model_f}} {env_dir} {exp_list} -o {env_dir}/data/experiment_lists/{exp_list}/trained_on_another_experiment_list/{train_exp_list}/{algo_config}'.format(
                env_dir=ENV_DIR,
                project_dir=PROJECT_DIR,
                exp_list='test_set.untampered_bulk_primary_cells_with_data',
                train_exp_list='train_set.untampered_bulk_primary_cells_with_data',
                algo_config=ISOTONIC_LOGISTIC_L2_LOG_CPM_ASSERT_AMBIG_NEG_FULL_DOWNWEIGHT
            )
        ]
        for c in commands:
            shell(c)    

rule apply_trained_model_to_single_cell_isotonic_logistic_regression_l2_assert_ambig_neg_downweight_by_group:
    input:
        test_exp_list = '{env_dir}/data/experiment_lists/{exp_list}/experiment_list.json'.format(
            env_dir=ENV_DIR,
            exp_list='untampered_single_cell_primary_cells_with_data'
        ),
        labelling_f = '{env_dir}/data/experiment_lists/{exp_list}/labelling.json'.format(
            env_dir=ENV_DIR,
            exp_list='untampered_single_cell_primary_cells_with_data'
        ),
        model_f = '{env_dir}/data/experiment_lists/{train_exp_list}/trained_models/{algo_config}/model.pickle'.format(
            env_dir=ENV_DIR,
            train_exp_list='untampered_bulk_primary_cells_with_data',
            algo_config=ISOTONIC_LOGISTIC_L2_LOG_CPM_ASSERT_AMBIG_NEG_FULL_DOWNWEIGHT
        )
    output:
        '{env_dir}/data/experiment_lists/{exp_list}/trained_on_another_experiment_list/{train_exp_list}/{algo_config}/prediction_results.json'.format(
            env_dir=ENV_DIR,
            exp_list='untampered_single_cell_primary_cells_with_data',
            train_exp_list='untampered_bulk_primary_cells_with_data',
            algo_config=ISOTONIC_LOGISTIC_L2_LOG_CPM_ASSERT_AMBIG_NEG_FULL_DOWNWEIGHT
        )
    run:
        commands = [
            'mkdir -p {env_dir}/data/experiment_lists/{exp_list}/trained_on_another_experiment_list/{train_exp_list}/{algo_config}'.format(
                env_dir=ENV_DIR,
                exp_list='untampered_single_cell_primary_cells_with_data',
                train_exp_list='untampered_bulk_primary_cells_with_data',
                algo_config=ISOTONIC_LOGISTIC_L2_LOG_CPM_ASSERT_AMBIG_NEG_FULL_DOWNWEIGHT
            ),
            'python2.7 {project_dir}/learning/apply_saved_model.py {{input.model_f}} {env_dir} {exp_list} -o {env_dir}/data/experiment_lists/{exp_list}/trained_on_another_experiment_list/{train_exp_list}/{algo_config}'.format(
                env_dir=ENV_DIR,
                project_dir=PROJECT_DIR,
                exp_list='untampered_single_cell_primary_cells_with_data',
                train_exp_list='untampered_bulk_primary_cells_with_data',
                algo_config=ISOTONIC_LOGISTIC_L2_LOG_CPM_ASSERT_AMBIG_NEG_FULL_DOWNWEIGHT
            )
        ]
        for c in commands:
            shell(c)

rule apply_trained_model_to_bulk_restricted_single_cell_isotonic_logistic_regression_l2_assert_ambig_neg_downweight_by_group:
    input:
        test_exp_list = '{env_dir}/data/experiment_lists/{exp_list}/experiment_list.json'.format(
            env_dir=ENV_DIR,
            exp_list='untampered_single_cell_primary_cells_with_data_cell_types_in_bulk'
        ),
        labelling_f = '{env_dir}/data/experiment_lists/{exp_list}/labelling.json'.format(
            env_dir=ENV_DIR,
            exp_list='untampered_single_cell_primary_cells_with_data_cell_types_in_bulk'
        ),
        model_f = '{env_dir}/data/experiment_lists/{train_exp_list}/trained_models/{algo_config}/model.pickle'.format(
            env_dir=ENV_DIR,
            train_exp_list='untampered_bulk_primary_cells_with_data',
            algo_config=ISOTONIC_LOGISTIC_L2_LOG_CPM_ASSERT_AMBIG_NEG_FULL_DOWNWEIGHT
        )
    output:
        '{env_dir}/data/experiment_lists/{exp_list}/trained_on_another_experiment_list/{train_exp_list}/{algo_config}/prediction_results.json'.format(
            env_dir=ENV_DIR,
            exp_list='untampered_single_cell_primary_cells_with_data_cell_types_in_bulk',
            train_exp_list='untampered_bulk_primary_cells_with_data',
            algo_config=ISOTONIC_LOGISTIC_L2_LOG_CPM_ASSERT_AMBIG_NEG_FULL_DOWNWEIGHT
        )
    run:
        commands = [
            'mkdir -p {env_dir}/data/experiment_lists/{exp_list}/trained_on_another_experiment_list/{train_exp_list}/{algo_config}'.format(
                env_dir=ENV_DIR,
                exp_list='untampered_single_cell_primary_cells_with_data_cell_types_in_bulk',
                train_exp_list='untampered_bulk_primary_cells_with_data',
                algo_config=ISOTONIC_LOGISTIC_L2_LOG_CPM_ASSERT_AMBIG_NEG_FULL_DOWNWEIGHT
            ),
            'python2.7 {project_dir}/learning/apply_saved_model.py {{input.model_f}} {env_dir} {exp_list} -o {env_dir}/data/experiment_lists/{exp_list}/trained_on_another_experiment_list/{train_exp_list}/{algo_config}'.format(
                env_dir=ENV_DIR,
                project_dir=PROJECT_DIR,
                exp_list='untampered_single_cell_primary_cells_with_data_cell_types_in_bulk',
                train_exp_list='untampered_bulk_primary_cells_with_data',
                algo_config=ISOTONIC_LOGISTIC_L2_LOG_CPM_ASSERT_AMBIG_NEG_FULL_DOWNWEIGHT
            )
        ]
        for c in commands:
            shell(c)

rule apply_trained_model_bnc_linear_svm_count_cond_prior_condor:
    input:
        test_exp_list = '{env_dir}/data/experiment_lists/{exp_list}/experiment_list.json'.format(
            env_dir=ENV_DIR,
            exp_list='test_set.untampered_bulk_primary_cells_with_data'
        ),
        labelling_f = '{env_dir}/data/experiment_lists/{exp_list}/labelling.json'.format(
            env_dir=ENV_DIR,
            exp_list='test_set.untampered_bulk_primary_cells_with_data'
        ),
        model_f = '{env_dir}/data/experiment_lists/{train_exp_list}/trained_models/{algo_config}/model.pickle'.format(
            env_dir=ENV_DIR,
            train_exp_list='train_set.untampered_bulk_primary_cells_with_data',
            algo_config=BNC_LOG_CPM_LINEAR_SVM_COND
        )
    output:
        '{env_dir}/data/experiment_lists/{exp_list}/trained_on_another_experiment_list/{train_exp_list}/{algo_config}/prediction_results.json'.format(
            env_dir=ENV_DIR,
            exp_list='test_set.untampered_bulk_primary_cells_with_data',
            train_exp_list='train_set.untampered_bulk_primary_cells_with_data',
            algo_config=BNC_LOG_CPM_LINEAR_SVM_COND
        )
    run:
        commands = [
            'mkdir -p {env_dir}/data/experiment_lists/{exp_list}/trained_on_another_experiment_list/{train_exp_list}/{algo_config}'.format(
                env_dir=ENV_DIR,
                exp_list='test_set.untampered_bulk_primary_cells_with_data',
                train_exp_list='train_set.untampered_bulk_primary_cells_with_data',
                algo_config=BNC_LOG_CPM_LINEAR_SVM_COND
            ),
            'python2.7 {project_dir}/learning/apply_saved_model_condorized/apply_saved_model_condor_primary.py {{input.model_f}} {env_dir} {exp_list} 5 -o {{output}} -c /scratch/mnbernstein/apply_trained_models_condor/{exp_list}.{algo_config}'.format(
                env_dir=ENV_DIR,
                project_dir=PROJECT_DIR,
                exp_list='test_set.untampered_bulk_primary_cells_with_data',
                train_exp_list='train_set.untampered_bulk_primary_cells_with_data',
                algo_config=BNC_LOG_CPM_LINEAR_SVM_COND
            )
        ]
        for c in commands:
            shell(c)

rule apply_trained_model_to_single_cell_bnc_linear_svm_count_cond_prior_condor:
    input:
        test_exp_list = '{env_dir}/data/experiment_lists/{exp_list}/experiment_list.json'.format(
            env_dir=ENV_DIR,
            exp_list='untampered_single_cell_primary_cells_with_data'
        ),
        labelling_f = '{env_dir}/data/experiment_lists/{exp_list}/labelling.json'.format(
            env_dir=ENV_DIR,
            exp_list='untampered_single_cell_primary_cells_with_data'
        ),
        model_f = '{env_dir}/data/experiment_lists/{train_exp_list}/trained_models/{algo_config}/model.pickle'.format(
            env_dir=ENV_DIR,
            train_exp_list='untampered_bulk_primary_cells_with_data',
            algo_config=BNC_LOG_CPM_LINEAR_SVM_COND
        )
    output:
        '{env_dir}/data/experiment_lists/{exp_list}/trained_on_another_experiment_list/{train_exp_list}/{algo_config}/prediction_results.json'.format(
            env_dir=ENV_DIR,
            exp_list='untampered_single_cell_primary_cells_with_data',
            train_exp_list='untampered_bulk_primary_cells_with_data',
            algo_config=BNC_LOG_CPM_LINEAR_SVM_COND
        )
    run:
        commands = [
            'mkdir -p {env_dir}/data/experiment_lists/{exp_list}/trained_on_another_experiment_list/{train_exp_list}/{algo_config}'.format(
                env_dir=ENV_DIR,
                exp_list='untampered_single_cell_primary_cells_with_data',
                train_exp_list='untampered_bulk_primary_cells_with_data',
                algo_config=BNC_LOG_CPM_LINEAR_SVM_COND
            ),
            'python2.7 {project_dir}/learning/apply_saved_model_condorized/apply_saved_model_condor_primary.py {{input.model_f}} {env_dir} {exp_list} 5 -o {{output}} -c /scratch/mnbernstein/apply_trained_models_condor/{exp_list}.{algo_config}'.format(
                env_dir=ENV_DIR,
                project_dir=PROJECT_DIR,
                exp_list='untampered_single_cell_primary_cells_with_data',
                train_exp_list='untampered_bulk_primary_cells_with_data',
                algo_config=BNC_LOG_CPM_LINEAR_SVM_COND
            )
        ]
        for c in commands:
            shell(c)


rule apply_trained_model_bnc_linear_svm_count_cond_prior_assert_ambig_neg_condor:
    input:
        test_exp_list = '{env_dir}/data/experiment_lists/{exp_list}/experiment_list.json'.format(
            env_dir=ENV_DIR,
            exp_list='test_set.untampered_bulk_primary_cells_with_data'
        ),
        labelling_f = '{env_dir}/data/experiment_lists/{exp_list}/labelling.json'.format(
            env_dir=ENV_DIR,
            exp_list='test_set.untampered_bulk_primary_cells_with_data'
        ),
        model_f = '{env_dir}/data/experiment_lists/{train_exp_list}/trained_models/{algo_config}/model.pickle'.format(
            env_dir=ENV_DIR,
            train_exp_list='train_set.untampered_bulk_primary_cells_with_data',
            algo_config=BNC_LOG_CPM_LINEAR_SVM_COND_ASSERT_AMBIG_NEG
        )
    output:
        '{env_dir}/data/experiment_lists/{exp_list}/trained_on_another_experiment_list/{train_exp_list}/{algo_config}/prediction_results.json'.format(
            env_dir=ENV_DIR,
            exp_list='test_set.untampered_bulk_primary_cells_with_data',
            train_exp_list='train_set.untampered_bulk_primary_cells_with_data',
            algo_config=BNC_LOG_CPM_LINEAR_SVM_COND_ASSERT_AMBIG_NEG
        )
    run:
        commands = [
            'mkdir -p {env_dir}/data/experiment_lists/{exp_list}/trained_on_another_experiment_list/{train_exp_list}/{algo_config}'.format(
                env_dir=ENV_DIR,
                exp_list='test_set.untampered_bulk_primary_cells_with_data',
                train_exp_list='train_set.untampered_bulk_primary_cells_with_data',
                algo_config=BNC_LOG_CPM_LINEAR_SVM_COND_ASSERT_AMBIG_NEG
            ),
            'python2.7 {project_dir}/learning/apply_saved_model_condorized/apply_saved_model_condor_primary.py {{input.model_f}} {env_dir} {exp_list} 5 -o {{output}} -c /scratch/mnbernstein/apply_trained_models_condor/{exp_list}.{algo_config}'.format(
                env_dir=ENV_DIR,
                project_dir=PROJECT_DIR,
                exp_list='test_set.untampered_bulk_primary_cells_with_data',
                train_exp_list='train_set.untampered_bulk_primary_cells_with_data',
                algo_config=BNC_LOG_CPM_LINEAR_SVM_COND_ASSERT_AMBIG_NEG
            )
        ]
        for c in commands:
            shell(c)


# TODO This is the fall back HERE
rule apply_trained_model_bnc_linear_svm_count_cond_prior_assert_ambig_neg:
    input:
        test_exp_list = '{env_dir}/data/experiment_lists/{exp_list}/experiment_list.json'.format(
            env_dir=ENV_DIR,
            exp_list='test_set.untampered_bulk_primary_cells_with_data'
        ),
        labelling_f = '{env_dir}/data/experiment_lists/{exp_list}/labelling.json'.format(
            env_dir=ENV_DIR,
            exp_list='test_set.untampered_bulk_primary_cells_with_data'
        ),
        model_f = '{env_dir}/data/experiment_lists/{train_exp_list}/trained_models/{algo_config}/model.pickle'.format(
            env_dir=ENV_DIR,
            train_exp_list='train_set.untampered_bulk_primary_cells_with_data',
            algo_config=BNC_LOG_CPM_LINEAR_SVM_COND_ASSERT_AMBIG_NEG
        )
    output:
        '{env_dir}/data/experiment_lists/{exp_list}/trained_on_another_experiment_list/{train_exp_list}/{algo_config}/prediction_results.json'.format(
            env_dir=ENV_DIR,
            exp_list='test_set.untampered_bulk_primary_cells_with_data',
            train_exp_list='train_set.untampered_bulk_primary_cells_with_data',
            algo_config=BNC_LOG_CPM_LINEAR_SVM_COND_ASSERT_AMBIG_NEG
        )
    run:
        commands = [
            'mkdir -p {env_dir}/data/experiment_lists/{exp_list}/trained_on_another_experiment_list/{train_exp_list}/{algo_config}'.format(
                env_dir=ENV_DIR,
                exp_list='test_set.untampered_bulk_primary_cells_with_data',
                train_exp_list='train_set.untampered_bulk_primary_cells_with_data',
                algo_config=BNC_LOG_CPM_LINEAR_SVM_COND_ASSERT_AMBIG_NEG
            ),
            'python2.7 {project_dir}/learning/apply_saved_model.py {{input.model_f}} {env_dir} {exp_list} -o {env_dir}/data/experiment_lists/{exp_list}/trained_on_another_experiment_list/{train_exp_list}/{algo_config}'.format(
                env_dir=ENV_DIR,
                project_dir=PROJECT_DIR,
                exp_list='test_set.untampered_bulk_primary_cells_with_data',
                train_exp_list='train_set.untampered_bulk_primary_cells_with_data',
                algo_config=BNC_LOG_CPM_LINEAR_SVM_COND_ASSERT_AMBIG_NEG
            )
        ]
        for c in commands:
            shell(c)


rule apply_trained_model_bnc_linear_svm_normal_count_cond_prior_condor:
    input:
        test_exp_list = '{env_dir}/data/experiment_lists/{exp_list}/experiment_list.json'.format(
            env_dir=ENV_DIR,
            exp_list='test_set.untampered_bulk_primary_cells_with_data'
        ),
        labelling_f = '{env_dir}/data/experiment_lists/{exp_list}/labelling.json'.format(
            env_dir=ENV_DIR,
            exp_list='test_set.untampered_bulk_primary_cells_with_data'
        ),
        model_f = '{env_dir}/data/experiment_lists/{train_exp_list}/trained_models/{algo_config}/model.pickle'.format(
            env_dir=ENV_DIR,
            train_exp_list='train_set.untampered_bulk_primary_cells_with_data',
            algo_config=BNC_LOG_CPM_LINEAR_SVM_NORMAL
        )
    output:
        '{env_dir}/data/experiment_lists/{exp_list}/trained_on_another_experiment_list/{train_exp_list}/{algo_config}/prediction_results.json'.format(
            env_dir=ENV_DIR,
            exp_list='test_set.untampered_bulk_primary_cells_with_data',
            train_exp_list='train_set.untampered_bulk_primary_cells_with_data',
            algo_config=BNC_LOG_CPM_LINEAR_SVM_NORMAL
        )
    run:
        commands = [
            'mkdir -p {env_dir}/data/experiment_lists/{exp_list}/trained_on_another_experiment_list/{train_exp_list}/{algo_config}'.format(
                env_dir=ENV_DIR,
                exp_list='test_set.untampered_bulk_primary_cells_with_data',
                train_exp_list='train_set.untampered_bulk_primary_cells_with_data',
                algo_config=BNC_LOG_CPM_LINEAR_SVM_NORMAL
            ),
            'python2.7 {project_dir}/learning/apply_saved_model_condorized/apply_saved_model_condor_primary.py {{input.model_f}} {env_dir} {exp_list} 5 -o {{output}} -c /scratch/mnbernstein/apply_trained_models_condor/{exp_list}.{algo_config}'.format(
                env_dir=ENV_DIR,
                project_dir=PROJECT_DIR,
                exp_list='test_set.untampered_bulk_primary_cells_with_data',
                train_exp_list='train_set.untampered_bulk_primary_cells_with_data',
                algo_config=BNC_LOG_CPM_LINEAR_SVM_NORMAL
            )
        ]
        for c in commands:
            shell(c)

rule apply_trained_model_bnc_linear_svm_normal_fixed_std_count_cond_prior_condor:
    input:
        test_exp_list = '{env_dir}/data/experiment_lists/{exp_list}/experiment_list.json'.format(
            env_dir=ENV_DIR,
            exp_list='test_set.untampered_bulk_primary_cells_with_data'
        ),
        labelling_f = '{env_dir}/data/experiment_lists/{exp_list}/labelling.json'.format(
            env_dir=ENV_DIR,
            exp_list='test_set.untampered_bulk_primary_cells_with_data'
        ),
        model_f = '{env_dir}/data/experiment_lists/{train_exp_list}/trained_models/{algo_config}/model.pickle'.format(
            env_dir=ENV_DIR,
            train_exp_list='train_set.untampered_bulk_primary_cells_with_data',
            algo_config=BNC_LOG_CPM_LINEAR_SVM_NORMAL_FIXED_STD
        )
    output:
        '{env_dir}/data/experiment_lists/{exp_list}/trained_on_another_experiment_list/{train_exp_list}/{algo_config}/prediction_results.json'.format(
            env_dir=ENV_DIR,
            exp_list='test_set.untampered_bulk_primary_cells_with_data',
            train_exp_list='train_set.untampered_bulk_primary_cells_with_data',
            algo_config=BNC_LOG_CPM_LINEAR_SVM_NORMAL_FIXED_STD
        )
    run:
        commands = [
            'mkdir -p {env_dir}/data/experiment_lists/{exp_list}/trained_on_another_experiment_list/{train_exp_list}/{algo_config}'.format(
                env_dir=ENV_DIR,
                exp_list='test_set.untampered_bulk_primary_cells_with_data',
                train_exp_list='train_set.untampered_bulk_primary_cells_with_data',
                algo_config=BNC_LOG_CPM_LINEAR_SVM_NORMAL_FIXED_STD
            ),
            'python2.7 {project_dir}/learning/apply_saved_model_condorized/apply_saved_model_condor_primary.py {{input.model_f}} {env_dir} {exp_list} 5 -o {{output}} -c /scratch/mnbernstein/apply_trained_models_condor/{exp_list}.{algo_config}'.format(
                env_dir=ENV_DIR,
                project_dir=PROJECT_DIR,
                exp_list='test_set.untampered_bulk_primary_cells_with_data',
                train_exp_list='train_set.untampered_bulk_primary_cells_with_data',
                algo_config=BNC_LOG_CPM_LINEAR_SVM_NORMAL_FIXED_STD
            )
        ]
        for c in commands:
            shell(c)

rule apply_trained_model_bnc_linear_logistic_bins_counts_prior_condor_downweight:
    input:
        test_exp_list = '{env_dir}/data/experiment_lists/{exp_list}/experiment_list.json'.format(
            env_dir=ENV_DIR,
            exp_list='test_set.untampered_bulk_primary_cells_with_data'
        ),
        labelling_f = '{env_dir}/data/experiment_lists/{exp_list}/labelling.json'.format(
            env_dir=ENV_DIR,
            exp_list='test_set.untampered_bulk_primary_cells_with_data'
        ),
        model_f = '{env_dir}/data/experiment_lists/{train_exp_list}/trained_models/{algo_config}/model.pickle'.format(
            env_dir=ENV_DIR,
            train_exp_list='train_set.untampered_bulk_primary_cells_with_data',
            algo_config=BNC_LOG_CPM_LOGISTIC_L2_DOWNWEIGHT
        )
    output:
        '{env_dir}/data/experiment_lists/{exp_list}/trained_on_another_experiment_list/{train_exp_list}/{algo_config}/prediction_results.json'.format(
            env_dir=ENV_DIR,
            exp_list='test_set.untampered_bulk_primary_cells_with_data',
            train_exp_list='train_set.untampered_bulk_primary_cells_with_data',
            algo_config=BNC_LOG_CPM_LOGISTIC_L2_DOWNWEIGHT
        )
    run:
        commands = [
            'mkdir -p {env_dir}/data/experiment_lists/{exp_list}/trained_on_another_experiment_list/{train_exp_list}/{algo_config}'.format(
                env_dir=ENV_DIR,
                exp_list='test_set.untampered_bulk_primary_cells_with_data',
                train_exp_list='train_set.untampered_bulk_primary_cells_with_data',
                algo_config=BNC_LOG_CPM_LOGISTIC_L2_DOWNWEIGHT
            ),
            'python2.7 {project_dir}/learning/apply_saved_model_condorized/apply_saved_model_condor_primary.py {{input.model_f}} {env_dir} {exp_list} 5 -o {{output}} -c /scratch/mnbernstein/apply_trained_models_condor/{exp_list}.{algo_config}'.format(
                env_dir=ENV_DIR,
                project_dir=PROJECT_DIR,
                exp_list='test_set.untampered_bulk_primary_cells_with_data',
                train_exp_list='train_set.untampered_bulk_primary_cells_with_data',
                algo_config=BNC_LOG_CPM_LOGISTIC_L2_DOWNWEIGHT
            )
        ]
        for c in commands:
            shell(c)

rule apply_trained_model_bnc_linear_logistic_bins_conditional_counts_prior_condor_downweight:
    input:
        test_exp_list = '{env_dir}/data/experiment_lists/{exp_list}/experiment_list.json'.format(
            env_dir=ENV_DIR,
            exp_list='test_set.untampered_bulk_primary_cells_with_data'
        ),
        labelling_f = '{env_dir}/data/experiment_lists/{exp_list}/labelling.json'.format(
            env_dir=ENV_DIR,
            exp_list='test_set.untampered_bulk_primary_cells_with_data'
        ),
        model_f = '{env_dir}/data/experiment_lists/{train_exp_list}/trained_models/{algo_config}/model.pickle'.format(
            env_dir=ENV_DIR,
            train_exp_list='train_set.untampered_bulk_primary_cells_with_data',
            algo_config=BNC_LOG_CPM_LOGISTIC_L2_COND_DOWNDWIEGHT
        )
    output:
        '{env_dir}/data/experiment_lists/{exp_list}/trained_on_another_experiment_list/{train_exp_list}/{algo_config}/prediction_results.json'.format(
            env_dir=ENV_DIR,
            exp_list='test_set.untampered_bulk_primary_cells_with_data',
            train_exp_list='train_set.untampered_bulk_primary_cells_with_data',
            algo_config=BNC_LOG_CPM_LOGISTIC_L2_COND_DOWNDWIEGHT
        )
    run:
        commands = [
            'mkdir -p {env_dir}/data/experiment_lists/{exp_list}/trained_on_another_experiment_list/{train_exp_list}/{algo_config}'.format(
                env_dir=ENV_DIR,
                exp_list='test_set.untampered_bulk_primary_cells_with_data',
                train_exp_list='train_set.untampered_bulk_primary_cells_with_data',
                algo_config=BNC_LOG_CPM_LOGISTIC_L2_COND_DOWNDWIEGHT
            ),
            'python2.7 {project_dir}/learning/apply_saved_model_condorized/apply_saved_model_condor_primary.py {{input.model_f}} {env_dir} {exp_list} 5 -o {{output}} -c /scratch/mnbernstein/apply_trained_models_condor/{exp_list}.{algo_config}'.format(
                env_dir=ENV_DIR,
                project_dir=PROJECT_DIR,
                exp_list='test_set.untampered_bulk_primary_cells_with_data',
                train_exp_list='train_set.untampered_bulk_primary_cells_with_data',
                algo_config=BNC_LOG_CPM_LOGISTIC_L2_COND_DOWNDWIEGHT
            )
        ]
        for c in commands:
            shell(c)

rule apply_trained_model_bnc_logistic_bins_counts_prior_condor:
    input:
        test_exp_list = '{env_dir}/data/experiment_lists/{exp_list}/experiment_list.json'.format(
            env_dir=ENV_DIR,
            exp_list='test_set.untampered_bulk_primary_cells_with_data'
        ),
        labelling_f = '{env_dir}/data/experiment_lists/{exp_list}/labelling.json'.format(
            env_dir=ENV_DIR,
            exp_list='test_set.untampered_bulk_primary_cells_with_data'
        ),
        model_f = '{env_dir}/data/experiment_lists/{train_exp_list}/trained_models/{algo_config}/model.pickle'.format(
            env_dir=ENV_DIR,
            train_exp_list='train_set.untampered_bulk_primary_cells_with_data',
            algo_config=BNC_LOG_CPM_LOGISTIC_L2
        )
    output:
        '{env_dir}/data/experiment_lists/{exp_list}/trained_on_another_experiment_list/{train_exp_list}/{algo_config}/prediction_results.json'.format(
            env_dir=ENV_DIR,
            exp_list='test_set.untampered_bulk_primary_cells_with_data',
            train_exp_list='train_set.untampered_bulk_primary_cells_with_data',
            algo_config=BNC_LOG_CPM_LOGISTIC_L2
        )
    run:
        commands = [
            'mkdir -p {env_dir}/data/experiment_lists/{exp_list}/trained_on_another_experiment_list/{train_exp_list}/{algo_config}'.format(
                env_dir=ENV_DIR,
                exp_list='test_set.untampered_bulk_primary_cells_with_data',
                train_exp_list='train_set.untampered_bulk_primary_cells_with_data',
                algo_config=BNC_LOG_CPM_LOGISTIC_L2
            ),
            'mkdir -p /scratch/mnbernstein/apply_trained_models_condor/{exp_list}.{algo_config}'.format(
                exp_list='test_set.untampered_bulk_primary_cells_with_data',
                algo_config=BNC_LOG_CPM_LOGISTIC_L2
            ),
            'python2.7 {project_dir}/learning/apply_saved_model_condorized/apply_saved_model_condor_primary.py {{input.model_f}} {env_dir} {exp_list} 5 -o {{output}} -c /scratch/mnbernstein/apply_trained_models_condor/{exp_list}.{algo_config}'.format(
                env_dir=ENV_DIR,
                project_dir=PROJECT_DIR,
                exp_list='test_set.untampered_bulk_primary_cells_with_data',
                train_exp_list='train_set.untampered_bulk_primary_cells_with_data',
                algo_config=BNC_LOG_CPM_LOGISTIC_L2
            )
        ]
        for c in commands:
            shell(c)
 
rule apply_trained_model_bnc_logistic_bins_cond_prior_condor:
    input:
        test_exp_list = '{env_dir}/data/experiment_lists/{exp_list}/experiment_list.json'.format(
            env_dir=ENV_DIR,
            exp_list='test_set.untampered_bulk_primary_cells_with_data'
        ),
        labelling_f = '{env_dir}/data/experiment_lists/{exp_list}/labelling.json'.format(
            env_dir=ENV_DIR,
            exp_list='test_set.untampered_bulk_primary_cells_with_data'
        ),
        model_f = '{env_dir}/data/experiment_lists/{train_exp_list}/trained_models/{algo_config}/model.pickle'.format(
            env_dir=ENV_DIR,
            train_exp_list='train_set.untampered_bulk_primary_cells_with_data',
            algo_config=BNC_LOG_CPM_LOGISTIC_L2_COND
        )
    output:
        '{env_dir}/data/experiment_lists/{exp_list}/trained_on_another_experiment_list/{train_exp_list}/{algo_config}/prediction_results.json'.format(
            env_dir=ENV_DIR,
            exp_list='test_set.untampered_bulk_primary_cells_with_data',
            train_exp_list='train_set.untampered_bulk_primary_cells_with_data',
            algo_config=BNC_LOG_CPM_LOGISTIC_L2_COND
        )
    run:
        commands = [
            'mkdir -p {env_dir}/data/experiment_lists/{exp_list}/trained_on_another_experiment_list/{train_exp_list}/{algo_config}'.format(
                env_dir=ENV_DIR,
                exp_list='test_set.untampered_bulk_primary_cells_with_data',
                train_exp_list='train_set.untampered_bulk_primary_cells_with_data',
                algo_config=BNC_LOG_CPM_LOGISTIC_L2_COND
            ),
            'mkdir -p /scratch/mnbernstein/apply_trained_models_condor/{exp_list}.{algo_config}'.format(
                exp_list='test_set.untampered_bulk_primary_cells_with_data',
                algo_config=BNC_LOG_CPM_LOGISTIC_L2_COND
            ),
            'python2.7 {project_dir}/learning/apply_saved_model_condorized/apply_saved_model_condor_primary.py {{input.model_f}} {env_dir} {exp_list} 5 -o {{output}} -c /scratch/mnbernstein/apply_trained_models_condor/{exp_list}.{algo_config}'.format(
                env_dir=ENV_DIR,
                project_dir=PROJECT_DIR,
                exp_list='test_set.untampered_bulk_primary_cells_with_data',
                train_exp_list='train_set.untampered_bulk_primary_cells_with_data',
                algo_config=BNC_LOG_CPM_LOGISTIC_L2_COND
            )
        ]
        for c in commands:
            shell(c)

   

rule apply_trained_model_tpr_logistic_regression_l2_assert_ambig_neg:
    input:
        test_exp_list = '{env_dir}/data/experiment_lists/{exp_list}/experiment_list.json'.format(
            env_dir=ENV_DIR,
            exp_list='test_set.untampered_bulk_primary_cells_with_data'
        ),
        labelling_f = '{env_dir}/data/experiment_lists/{exp_list}/labelling.json'.format(
            env_dir=ENV_DIR,
            exp_list='test_set.untampered_bulk_primary_cells_with_data'
        ),
        model_f = '{env_dir}/data/experiment_lists/{train_exp_list}/trained_models/{algo_config}/model.pickle'.format(
            env_dir=ENV_DIR,
            train_exp_list='train_set.untampered_bulk_primary_cells_with_data',
            algo_config=TPR_LOGISTIC_L2_LOG_CPM_FULL
        )
    output:
        '{env_dir}/data/experiment_lists/{exp_list}/trained_on_another_experiment_list/{train_exp_list}/{algo_config}/prediction_results.json'.format(
            env_dir=ENV_DIR,
            exp_list='test_set.untampered_bulk_primary_cells_with_data',
            train_exp_list='train_set.untampered_bulk_primary_cells_with_data',
            algo_config=TPR_LOGISTIC_L2_LOG_CPM_FULL
        )
    run:
        commands = [
            'mkdir -p {env_dir}/data/experiment_lists/{exp_list}/trained_on_another_experiment_list/{train_exp_list}/{algo_config}'.format(
                env_dir=ENV_DIR,
                exp_list='test_set.untampered_bulk_primary_cells_with_data',
                train_exp_list='train_set.untampered_bulk_primary_cells_with_data',
                algo_config=TPR_LOGISTIC_L2_LOG_CPM_FULL
            ),
            'python2.7 {project_dir}/learning/apply_saved_model.py {{input.model_f}} {env_dir} {exp_list} -o {env_dir}/data/experiment_lists/{exp_list}/trained_on_another_experiment_list/{train_exp_list}/{algo_config}'.format(
                env_dir=ENV_DIR,
                project_dir=PROJECT_DIR,
                exp_list='test_set.untampered_bulk_primary_cells_with_data',
                train_exp_list='train_set.untampered_bulk_primary_cells_with_data',
                algo_config=TPR_LOGISTIC_L2_LOG_CPM_FULL
            )
        ]
        for c in commands:
            shell(c)

rule apply_trained_model_to_single_cell_tpr_logistic_regression_l2_assert_ambig_neg:
    input:
        test_exp_list = '{env_dir}/data/experiment_lists/{exp_list}/experiment_list.json'.format(
            env_dir=ENV_DIR,
            exp_list='untampered_single_cell_primary_cells_with_data'
        ),
        labelling_f = '{env_dir}/data/experiment_lists/{exp_list}/labelling.json'.format(
            env_dir=ENV_DIR,
            exp_list='untampered_single_cell_primary_cells_with_data'
        ),
        model_f = '{env_dir}/data/experiment_lists/{train_exp_list}/trained_models/{algo_config}/model.pickle'.format(
            env_dir=ENV_DIR,
            train_exp_list='untampered_bulk_primary_cells_with_data',
            algo_config=TPR_LOGISTIC_L2_LOG_CPM_FULL
        )
    output:
        '{env_dir}/data/experiment_lists/{exp_list}/trained_on_another_experiment_list/{train_exp_list}/{algo_config}/prediction_results.json'.format(
            env_dir=ENV_DIR,
            exp_list='untampered_single_cell_primary_cells_with_data',
            train_exp_list='untampered_bulk_primary_cells_with_data',
            algo_config=TPR_LOGISTIC_L2_LOG_CPM_FULL
        )
    run:
        commands = [
            'mkdir -p {env_dir}/data/experiment_lists/{exp_list}/trained_on_another_experiment_list/{train_exp_list}/{algo_config}'.format(
                env_dir=ENV_DIR,
                exp_list='untampered_single_cell_primary_cells_with_data',
                train_exp_list='untampered_bulk_primary_cells_with_data',
                algo_config=TPR_LOGISTIC_L2_LOG_CPM_FULL
            ),
            'python2.7 {project_dir}/learning/apply_saved_model.py {{input.model_f}} {env_dir} {exp_list} -o {env_dir}/data/experiment_lists/{exp_list}/trained_on_another_experiment_list/{train_exp_list}/{algo_config}'.format(
                env_dir=ENV_DIR,
                project_dir=PROJECT_DIR,
                exp_list='untampered_single_cell_primary_cells_with_data',
                train_exp_list='untampered_bulk_primary_cells_with_data',
                algo_config=TPR_LOGISTIC_L2_LOG_CPM_FULL
            )
        ]
        for c in commands:
            shell(c)


rule apply_trained_model_tpr_logistic_regression_l2_assert_ambig_neg_downweight_by_group:
    input:
        test_exp_list = '{env_dir}/data/experiment_lists/{exp_list}/experiment_list.json'.format(
            env_dir=ENV_DIR,
            exp_list='test_set.untampered_bulk_primary_cells_with_data'
        ),
        labelling_f = '{env_dir}/data/experiment_lists/{exp_list}/labelling.json'.format(
            env_dir=ENV_DIR,
            exp_list='test_set.untampered_bulk_primary_cells_with_data'
        ),
        model_f = '{env_dir}/data/experiment_lists/{train_exp_list}/trained_models/{algo_config}/model.pickle'.format(
            env_dir=ENV_DIR,
            train_exp_list='train_set.untampered_bulk_primary_cells_with_data',
            algo_config=TPR_LOGISTIC_L2_LOG_CPM_FULL_DOWNWEIGHT
        )
    output:
        '{env_dir}/data/experiment_lists/{exp_list}/trained_on_another_experiment_list/{train_exp_list}/{algo_config}/prediction_results.json'.format(
            env_dir=ENV_DIR,
            exp_list='test_set.untampered_bulk_primary_cells_with_data',
            train_exp_list='train_set.untampered_bulk_primary_cells_with_data',
            algo_config=TPR_LOGISTIC_L2_LOG_CPM_FULL_DOWNWEIGHT
        )
    run:
        commands = [
            'mkdir -p {env_dir}/data/experiment_lists/{exp_list}/trained_on_another_experiment_list/{train_exp_list}/{algo_config}'.format(
                env_dir=ENV_DIR,
                exp_list='test_set.untampered_bulk_primary_cells_with_data',
                train_exp_list='train_set.untampered_bulk_primary_cells_with_data',
                algo_config=TPR_LOGISTIC_L2_LOG_CPM_FULL_DOWNWEIGHT
            ),
            'python2.7 {project_dir}/learning/apply_saved_model.py {{input.model_f}} {env_dir} {exp_list} -o {env_dir}/data/experiment_lists/{exp_list}/trained_on_another_experiment_list/{train_exp_list}/{algo_config}'.format(
                env_dir=ENV_DIR,
                project_dir=PROJECT_DIR,
                exp_list='test_set.untampered_bulk_primary_cells_with_data',
                train_exp_list='train_set.untampered_bulk_primary_cells_with_data',
                algo_config=TPR_LOGISTIC_L2_LOG_CPM_FULL_DOWNWEIGHT
            )
        ]
        for c in commands:
            shell(c)

rule apply_trained_model_to_single_cell_tpr_logistic_regression_l2_assert_ambig_neg_downweight_by_group:
    input:
        test_exp_list = '{env_dir}/data/experiment_lists/{exp_list}/experiment_list.json'.format(
            env_dir=ENV_DIR,
            exp_list='untampered_single_cell_primary_cells_with_data'
        ),
        labelling_f = '{env_dir}/data/experiment_lists/{exp_list}/labelling.json'.format(
            env_dir=ENV_DIR,
            exp_list='untampered_single_cell_primary_cells_with_data'
        ),
        model_f = '{env_dir}/data/experiment_lists/{train_exp_list}/trained_models/{algo_config}/model.pickle'.format(
            env_dir=ENV_DIR,
            train_exp_list='untampered_bulk_primary_cells_with_data',
            algo_config=TPR_LOGISTIC_L2_LOG_CPM_FULL_DOWNWEIGHT
        )
    output:
        '{env_dir}/data/experiment_lists/{exp_list}/trained_on_another_experiment_list/{train_exp_list}/{algo_config}/prediction_results.json'.format(
            env_dir=ENV_DIR,
            exp_list='untampered_single_cell_primary_cells_with_data',
            train_exp_list='untampered_bulk_primary_cells_with_data',
            algo_config=TPR_LOGISTIC_L2_LOG_CPM_FULL_DOWNWEIGHT
        )
    run:
        commands = [
            'mkdir -p {env_dir}/data/experiment_lists/{exp_list}/trained_on_another_experiment_list/{train_exp_list}/{algo_config}'.format(
                env_dir=ENV_DIR,
                exp_list='untampered_single_cell_primary_cells_with_data',
                train_exp_list='untampered_bulk_primary_cells_with_data',
                algo_config=TPR_LOGISTIC_L2_LOG_CPM_FULL_DOWNWEIGHT
            ),
            'python2.7 {project_dir}/learning/apply_saved_model.py {{input.model_f}} {env_dir} {exp_list} -o {env_dir}/data/experiment_lists/{exp_list}/trained_on_another_experiment_list/{train_exp_list}/{algo_config}'.format(
                env_dir=ENV_DIR,
                project_dir=PROJECT_DIR,
                exp_list='untampered_single_cell_primary_cells_with_data',
                train_exp_list='untampered_bulk_primary_cells_with_data',
                algo_config=TPR_LOGISTIC_L2_LOG_CPM_FULL_DOWNWEIGHT
            )
        ]
        for c in commands:
            shell(c)

rule apply_trained_model_to_bulk_restricted_single_cell_tpr_logistic_regression_l2_assert_ambig_neg_downweight_by_group:
    input:
        test_exp_list = '{env_dir}/data/experiment_lists/{exp_list}/experiment_list.json'.format(
            env_dir=ENV_DIR,
            exp_list='untampered_single_cell_primary_cells_with_data_cell_types_in_bulk'
        ),
        labelling_f = '{env_dir}/data/experiment_lists/{exp_list}/labelling.json'.format(
            env_dir=ENV_DIR,
            exp_list='untampered_single_cell_primary_cells_with_data_cell_types_in_bulk'
        ),
        model_f = '{env_dir}/data/experiment_lists/{train_exp_list}/trained_models/{algo_config}/model.pickle'.format(
            env_dir=ENV_DIR,
            train_exp_list='untampered_bulk_primary_cells_with_data',
            algo_config=TPR_LOGISTIC_L2_LOG_CPM_FULL_DOWNWEIGHT
        )
    output:
        '{env_dir}/data/experiment_lists/{exp_list}/trained_on_another_experiment_list/{train_exp_list}/{algo_config}/prediction_results.json'.format(
            env_dir=ENV_DIR,
            exp_list='untampered_single_cell_primary_cells_with_data_cell_types_in_bulk',
            train_exp_list='untampered_bulk_primary_cells_with_data',
            algo_config=TPR_LOGISTIC_L2_LOG_CPM_FULL_DOWNWEIGHT
        )
    run:
        commands = [
            'mkdir -p {env_dir}/data/experiment_lists/{exp_list}/trained_on_another_experiment_list/{train_exp_list}/{algo_config}'.format(
                env_dir=ENV_DIR,
                exp_list='untampered_single_cell_primary_cells_with_data_cell_types_in_bulk',
                train_exp_list='untampered_bulk_primary_cells_with_data',
                algo_config=TPR_LOGISTIC_L2_LOG_CPM_FULL_DOWNWEIGHT
            ),
            'python2.7 {project_dir}/learning/apply_saved_model.py {{input.model_f}} {env_dir} {exp_list} -o {env_dir}/data/experiment_lists/{exp_list}/trained_on_another_experiment_list/{train_exp_list}/{algo_config}'.format(
                env_dir=ENV_DIR,
                project_dir=PROJECT_DIR,
                exp_list='untampered_single_cell_primary_cells_with_data_cell_types_in_bulk',
                train_exp_list='untampered_bulk_primary_cells_with_data',
                algo_config=TPR_LOGISTIC_L2_LOG_CPM_FULL_DOWNWEIGHT
            )
        ]
        for c in commands:
            shell(c)

rule train_and_apply_one_nn:
    input:
        '{env_dir}/data/experiment_lists/{exp_list}/experiment_list.json'.format(
            env_dir=ENV_DIR,
            exp_list='train_set.untampered_bulk_primary_cells_with_data'
        ),
        '{env_dir}/data/experiment_lists/{exp_list}/experiment_list.json'.format(
            env_dir=ENV_DIR,
            exp_list='test_set.untampered_bulk_primary_cells_with_data'
        ),
        '{env_dir}/data/experiment_lists/{exp_list}/labelling.json'.format(
            env_dir=ENV_DIR,
            exp_list='untampered_bulk_primary_cells_with_data'
        )
    output: 
        '{env_dir}/data/experiment_lists/{exp_list}/trained_on_another_experiment_list/{train_exp_list}/{algo_config}/prediction_results.json'.format(
            env_dir=ENV_DIR,
            exp_list='test_set.untampered_bulk_primary_cells_with_data',
            train_exp_list='train_set.untampered_bulk_primary_cells_with_data',
            algo_config=ONE_NN_LOG_CPM_CORRELATION
        )
    run:
        commands = [
            'mkdir -p /scratch/mnbernstein/cell_type_phenotyping_tmp/train_and_save_artifacts',
            'mkdir -p {env_dir}/data/experiment_lists/{exp_list}/trained_on_another_experiment_list/{train_exp_list}/{algo_config}'.format(
                env_dir=ENV_DIR,
                exp_list='test_set.untampered_bulk_primary_cells_with_data',
                train_exp_list='train_set.untampered_bulk_primary_cells_with_data',
                algo_config=ONE_NN_LOG_CPM_CORRELATION
            ),
            'python2.7 {project_dir}/learning/train_and_apply.py {project_dir}/learning/train_and_save_config/{env_name}/{algo_config}.json {env_dir} {train_exp_list} {exp_list} -r /scratch/mnbernstein/cell_type_phenotyping_tmp/train_and_save_artifacts -o {env_dir}/data/experiment_lists/{exp_list}/trained_on_another_experiment_list/{train_exp_list}/{algo_config}'.format(
                env_dir=ENV_DIR,
                env_name=ENV_NAME,
                project_dir=PROJECT_DIR,
                exp_list='test_set.untampered_bulk_primary_cells_with_data',
                train_exp_list='train_set.untampered_bulk_primary_cells_with_data',
                algo_config=ONE_NN_LOG_CPM_CORRELATION
            )

        ]
        for c in commands:
            shell(c)

rule train_and_apply_to_single_cell_one_nn:
    input:
        '{env_dir}/data/experiment_lists/{exp_list}/experiment_list.json'.format(
            env_dir=ENV_DIR,
            exp_list='untampered_bulk_primary_cells_with_data'
        ),
        '{env_dir}/data/experiment_lists/{exp_list}/experiment_list.json'.format(
            env_dir=ENV_DIR,
            exp_list='untampered_single_cell_primary_cells_with_data'
        ),
        '{env_dir}/data/experiment_lists/{exp_list}/labelling.json'.format(
            env_dir=ENV_DIR,
            exp_list='untampered_bulk_primary_cells_with_data'
        )
    output:
        '{env_dir}/data/experiment_lists/{exp_list}/trained_on_another_experiment_list/{train_exp_list}/{algo_config}/prediction_results.json'.format(
            env_dir=ENV_DIR,
            exp_list='untampered_single_cell_primary_cells_with_data',
            train_exp_list='untampered_bulk_primary_cells_with_data',
            algo_config=ONE_NN_LOG_CPM_CORRELATION
        )
    run:
        commands = [
            'mkdir -p /scratch/mnbernstein/cell_type_phenotyping_tmp/train_and_save_artifacts',
            'mkdir -p {env_dir}/data/experiment_lists/{exp_list}/trained_on_another_experiment_list/{train_exp_list}/{algo_config}'.format(
                env_dir=ENV_DIR,
                exp_list='untampered_single_cell_primary_cells_with_data',
                train_exp_list='untampered_bulk_primary_cells_with_data',
                algo_config=ONE_NN_LOG_CPM_CORRELATION
            ),
            'python2.7 {project_dir}/learning/train_and_apply.py {project_dir}/learning/train_and_save_config/{env_name}/{algo_config}.json {env_dir} {train_exp_list} {exp_list} -r /scratch/mnbernstein/cell_type_phenotyping_tmp/train_and_save_artifacts -o {env_dir}/data/experiment_lists/{exp_list}/trained_on_another_experiment_list/{train_exp_list}/{algo_config}'.format(
                env_dir=ENV_DIR,
                env_name=ENV_NAME,
                project_dir=PROJECT_DIR,
                exp_list='untampered_single_cell_primary_cells_with_data',
                train_exp_list='untampered_bulk_primary_cells_with_data',
                algo_config=ONE_NN_LOG_CPM_CORRELATION
            )

        ]
        for c in commands:
            shell(c)

rule train_and_apply_to_bulk_restricted_single_cell_one_nn:
    input:
        '{env_dir}/data/experiment_lists/{exp_list}/experiment_list.json'.format(
            env_dir=ENV_DIR,
            exp_list='untampered_bulk_primary_cells_with_data'
        ),
        '{env_dir}/data/experiment_lists/{exp_list}/experiment_list.json'.format(
            env_dir=ENV_DIR,
            exp_list='untampered_single_cell_primary_cells_with_data_cell_types_in_bulk'
        ),
        '{env_dir}/data/experiment_lists/{exp_list}/labelling.json'.format(
            env_dir=ENV_DIR,
            exp_list='untampered_bulk_primary_cells_with_data'
        )
    output:
        '{env_dir}/data/experiment_lists/{exp_list}/trained_on_another_experiment_list/{train_exp_list}/{algo_config}/prediction_results.json'.format(
            env_dir=ENV_DIR,
            exp_list='untampered_single_cell_primary_cells_with_data_cell_types_in_bulk',
            train_exp_list='untampered_bulk_primary_cells_with_data',
            algo_config=ONE_NN_LOG_CPM_CORRELATION
        )
    run:
        commands = [
            'mkdir -p /scratch/mnbernstein/cell_type_phenotyping_tmp/train_and_save_artifacts',
            'mkdir -p {env_dir}/data/experiment_lists/{exp_list}/trained_on_another_experiment_list/{train_exp_list}/{algo_config}'.format(
                env_dir=ENV_DIR,
                exp_list='untampered_single_cell_primary_cells_with_data_cell_types_in_bulk',
                train_exp_list='untampered_bulk_primary_cells_with_data',
                algo_config=ONE_NN_LOG_CPM_CORRELATION
            ),
            'python2.7 {project_dir}/learning/train_and_apply.py {project_dir}/learning/train_and_save_config/{env_name}/{algo_config}.json {env_dir} {train_exp_list} {exp_list} -r /scratch/mnbernstein/cell_type_phenotyping_tmp/train_and_save_artifacts -o {env_dir}/data/experiment_lists/{exp_list}/trained_on_another_experiment_list/{train_exp_list}/{algo_config}'.format(
                env_dir=ENV_DIR,
                env_name=ENV_NAME,
                project_dir=PROJECT_DIR,
                exp_list='untampered_single_cell_primary_cells_with_data_cell_types_in_bulk',
                train_exp_list='untampered_bulk_primary_cells_with_data',
                algo_config=ONE_NN_LOG_CPM_CORRELATION
            )

        ]
        for c in commands:
            shell(c)

#####################################################################
#   Compute data set summary statistics and create figures
#####################################################################

rule dim_reduc_overlayed_onto_ontology_figs:
    input:
        expand('{}/data/experiment_lists/{{exp_list}}/labelling.json'.format(ENV_DIR), exp_list=EXPERIMENT_LISTS),
        expand('{}/data/experiment_lists/{{exp_list}}/experiment_list.json'.format(ENV_DIR), exp_list=EXPERIMENT_LISTS)
    output:
        expand('{}/data/experiment_lists/{{exp_list}}/summary_figures/pca_color_by_child_on_graph.pdf'.format(ENV_DIR), exp_list=EXPERIMENT_LISTS)
    run:
        commands = expand(
            'python2.7 {project_dir}/manage_data/data_set_summary/dim_reduc_color_by_child_term.py {env_dir} {{exp_list}} -o {env_dir}/data/experiment_lists/{{exp_list}}/summary_figures'.format(
                project_dir=PROJECT_DIR,
                env_dir=ENV_DIR
            ),
            exp_list=EXPERIMENT_LISTS
        )
        for c in commands:
            shell(c)

rule compute_num_expressed_genes:
    input:
        expand('{}/data/experiment_lists/{{exp_list}}/experiment_list.json'.format(ENV_DIR), exp_list=EXPERIMENT_LISTS)
    output:
        expand('{}/data/experiment_lists/{{exp_list}}/summary_figures/experiment_to_num_genes_expressed.json'.format(ENV_DIR), exp_list=EXPERIMENT_LISTS)
    run:
        commands = expand(
            'python2.7 {project_dir}/manage_data/data_set_summary/compute_distr_num_genes_expressed.py {env_dir} {{exp_list}} -o {env_dir}/data/experiment_lists/{{exp_list}}/summary_figures'.format(
                project_dir=PROJECT_DIR,
                env_dir=ENV_DIR
            ),
            exp_list=EXPERIMENT_LISTS
        )
        for c in commands:
            shell(c)

rule num_expressed_genes_figs:
    input:
        expand('{}/data/experiment_lists/{{exp_list}}/summary_figures/experiment_to_num_genes_expressed.json'.format(ENV_DIR), exp_list=EXPERIMENT_LISTS)
    output:
        expand('{}/data/experiment_lists/{{exp_list}}/summary_figures/num_genes_expressed_distribution.pdf'.format(ENV_DIR), exp_list=EXPERIMENT_LISTS)
    run:
        commands = expand(
            'python2.7 {project_dir}/manage_data/data_set_summary/analyze_distr_num_genes_expressed.py {{input}} -o {env_dir}/data/experiment_lists/{{exp_list}}/summary_figures'.format(
                project_dir=PROJECT_DIR,
                env_dir=ENV_DIR
            ),
            exp_list=EXPERIMENT_LISTS
        )
        for c in commands:
            shell(c)



#####################################################################
#   Create the various experiment lists
#####################################################################

rule toy_single_cell_exp_list:
    input:
        '{}/data/data_set_metadata.json'.format(ENV_DIR)
    output:
        '{}/data/experiment_lists/toy_single_cell/experiment_list.json'.format(ENV_DIR)
    run:
        commands = [
            'mkdir -p {}/data/experiment_lists/toy_single_cell/summary_figures'.format(ENV_DIR),
            'mkdir -p {}/data/experiment_lists/toy_single_cell/leave_study_out_cv_results'.format(ENV_DIR),
            'python2.7 {}/manage_data/create_experiment_lists/cell_type.metasra_1-4/toy_single_cell.py {{input}} -o {{output}}'.format(
                PROJECT_DIR
            )
        ]
        for c in commands:
            shell(c)

rule toy_exp_list:
    input:
        '{}/data/data_set_metadata.json'.format(ENV_DIR)
    output:
        '{}/data/experiment_lists/toy/experiment_list.json'.format(ENV_DIR)
    run:
        commands = [
            'mkdir -p {}/data/experiment_lists/toy/summary_figures'.format(ENV_DIR),
            'mkdir -p {}/data/experiment_lists/toy/leave_study_out_cv_results'.format(ENV_DIR),
            'python2.7 {}/manage_data/create_experiment_lists/cell_type.metasra_1-4/toy.py {{input}} -o {{output}}'.format(
                PROJECT_DIR
            )
        ]
        for c in commands:
            shell(c)

rule test_exp_list:
    input:
        '{}/data/data_set_metadata.json'.format(ENV_DIR),
        '{}/data/experiment_lists/untampered_bulk_primary_cells_with_data/experiment_list.json'.format(ENV_DIR)
    output:
        '{}/data/experiment_lists/test_experiments/experiment_list.json'.format(ENV_DIR)
    run:
        commands = [
            'mkdir -p {}/data/experiment_lists/test_experiments/summary_figures'.format(ENV_DIR),
            'mkdir -p {}/data/experiment_lists/test_experiments/leave_study_out_cv_results'.format(ENV_DIR),
            'python2.7 {}/manage_data/create_experiment_lists/cell_type.metasra_1-4/test_experiments.py {{input}} -o {{output}}'.format(
                PROJECT_DIR
            )
        ]
        for c in commands:
            shell(c)   

rule train_test_experiment_lists:
    input:
        data_set='{}/data/data_set_metadata.json'.format(ENV_DIR),
        untampered_exp_list='{}/data/experiment_lists/all_untampered_bulk_primary_cells_with_data/experiment_list.json'.format(ENV_DIR),
    output:
        train_exp_list_f='{}/data/experiment_lists/train_set.untampered_bulk_primary_cells_with_data/experiment_list.json'.format(ENV_DIR),
        test_exp_list_f='{}/data/experiment_lists/test_set.untampered_bulk_primary_cells_with_data/experiment_list.json'.format(ENV_DIR)
    run:
        commands = [
            'mkdir -p {}/data/experiment_lists/train_set.untampered_bulk_primary_cells_with_data'.format(ENV_DIR),
            'mkdir -p {}/data/experiment_lists/test_set.untampered_bulk_primary_cells_with_data'.format(ENV_DIR),
            'python2.7 {project_dir}/manage_data/create_experiment_lists/cell_type.metasra_1-4/train_test_from_untampered_bulk_primary_cells_with_data.py {{input.data_set}} {{input.untampered_exp_list}} {project_dir}/manage_data/create_experiment_lists/cell_type.metasra_1-4/train_test_partition_data.untampered_bulk_primary_cells_with_data.v4.json -r {{output.train_exp_list_f}} -e {{output.test_exp_list_f}}'.format(project_dir=PROJECT_DIR)
        ]
        for c in commands:
            shell(c)

rule untampered_single_cell_primary_cells_with_data_cell_types_in_bulk_experiment_list:
    input:
        sc_exp_list='{}/data/experiment_lists/untampered_single_cell_primary_cells_with_data/experiment_list.json'.format(ENV_DIR),
        b_exp_list='{}/data/experiment_lists/untampered_bulk_primary_cells_with_data/experiment_list.json'.format(ENV_DIR),
        sc_graph='{}/data/experiment_lists/untampered_single_cell_primary_cells_with_data/labelling.json'.format(ENV_DIR),
        b_graph='{}/data/experiment_lists/untampered_bulk_primary_cells_with_data/labelling.json'.format(ENV_DIR)
    output:
        '{}/data/experiment_lists/untampered_single_cell_primary_cells_with_data_cell_types_in_bulk/experiment_list.json'.format(ENV_DIR)
    run:
        commands = [
            'mkdir -p {}/data/experiment_lists/untampered_single_cell_primary_cells_with_data_cell_types_in_bulk'.format(ENV_DIR),
            'python2.7 {project_dir}/manage_data/create_experiment_lists/cell_type.metasra_1-4/untampered_single_cell_primary_cells_with_data_cell_types_in_bulk.py {{input.sc_exp_list}} {{input.b_exp_list}} {{input.sc_graph}} {{input.b_graph}} -o {{output}}'.format(
                project_dir=PROJECT_DIR
            )
        ]
        for c in commands:
            shell(c)

rule untampered_cells_with_data_experiment_list:
    input:
        sc_exp_list='{}/data/experiment_lists/untampered_single_cell_primary_cells_with_data/experiment_list.json'.format(ENV_DIR),
        b_exp_list='{}/data/experiment_lists/untampered_bulk_primary_cells_with_data/experiment_list.json'.format(ENV_DIR)
    output:
        '{}/data/experiment_lists/untampered_cells_with_data/experiment_list.json'.format(ENV_DIR)
    run:
        commands = [
            'mkdir -p {}/data/experiment_lists/untampered_single_cell_primary_cells_with_data_cell_types_in_bulk'.format(ENV_DIR),
            'python2.7 {project_dir}/manage_data/create_experiment_lists/cell_type.metasra_1-4/untampered_cells_with_data.py {{input.sc_exp_list}} {{input.b_exp_list}} -o {{output}}'.format(
                project_dir=PROJECT_DIR
            )
        ]
        for c in commands:
            shell(c)

rule untampered_single_cell_primary_cells_with_data_list:
    input:
        '{}/data/data_set_metadata.json'.format(ENV_DIR)
    output:
        '{}/data/experiment_lists/untampered_single_cell_primary_cells_with_data/experiment_list.json'.format(ENV_DIR)
    run:
        commands = [
            'mkdir -p {}/data/experiment_lists/untampered_single_cell_primary_cells_with_data'.format(ENV_DIR),
            'python2.7 {project_dir}/manage_data/create_experiment_lists/cell_type.metasra_1-4/untampered_single_cell_primary_cells_with_data.py {annot_f} -o {{output}}'.format(
            project_dir=PROJECT_DIR,
            annot_f=ANNOTATION_F
            )
        ]
        for c in commands:
            shell(c)

rule single_cell_with_data_list:
    input:
        '{}/data/data_set_metadata.json'.format(ENV_DIR)
    output:
        '{}/data/experiment_lists/single_cell_with_data/experiment_list.json'.format(ENV_DIR)
    run:
        commands = [
            'mkdir -p {}/data/experiment_lists/single_cell_with_data'.format(ENV_DIR),
            'python2.7 {project_dir}/manage_data/create_experiment_lists/cell_type.metasra_1-4/single_cell_with_data.py {annot_f} -o {{output}}'.format(
            project_dir=PROJECT_DIR,
            annot_f=ANNOTATION_F
            )
        ]
        for c in commands:
            shell(c)

rule union_train_test_untampered_bulk_exps_w_data_list:
    input:
        train_exp_list_f='{}/data/experiment_lists/train_set.untampered_bulk_primary_cells_with_data/experiment_list.json'.format(ENV_DIR),
        test_exp_list_f='{}/data/experiment_lists/test_set.untampered_bulk_primary_cells_with_data/experiment_list.json'.format(ENV_DIR)
    output:
        '{}/data/experiment_lists/untampered_bulk_primary_cells_with_data/experiment_list.json'.format(ENV_DIR)
    run:
        commands = [
            'mkdir -p {}/data/experiment_lists/untampered_bulk_primary_cells_with_data'.format(ENV_DIR),
            'python2.7 {project_dir}/manage_data/create_experiment_lists/cell_type.metasra_1-4/union_train_test.py {{input.train_exp_list_f}} {{input.test_exp_list_f}} -o {{output}}'.format(
                project_dir=PROJECT_DIR
            )
        ]
        for c in commands:
            shell(c)

rule all_untampered_bulk_exps_w_data_list:
    input:
        '{}/data/data_set_metadata.json'.format(ENV_DIR)
    output:
        '{}/data/experiment_lists/all_untampered_bulk_primary_cells_with_data/experiment_list.json'.format(ENV_DIR)
    run:
        commands = [
            'mkdir -p {}/data/experiment_lists/all_untampered_bulk_primary_cells_with_data'.format(ENV_DIR),
            'python2.7 {project_dir}/manage_data/create_experiment_lists/cell_type.metasra_1-4/all_untampered_bulk_primary_cells_with_data.py {annot_f} -o {{output}}'.format(
                project_dir=PROJECT_DIR,
                annot_f=ANNOTATION_F
            )
        ]
        for c in commands:
            shell(c)

rule all_exps_w_data_list:
    input:
        '{}/data/data_set_metadata.json'.format(ENV_DIR)
    output:
        '{}/data/experiment_lists/all_experiments_with_data/experiment_list.json'.format(ENV_DIR)
    run:
        commands = [
            'mkdir -p {}/data/experiment_lists/all_experiments_with_data'.format(ENV_DIR),
            'python2.7 {}/manage_data/create_experiment_lists/cell_type.metasra_1-4/all_experiments_with_data.py {{input}} -o {{output}}'.format(PROJECT_DIR)
        ]
        for c in commands:
            shell(c)


#####################################################################
#   Create the label graphs
#####################################################################

rule generate_label_graphs:
    input:
        expand('{}/data/experiment_lists/{{exp_list}}/experiment_list.json'.format(ENV_DIR), exp_list=EXPERIMENT_LISTS)
    output:
        expand('{}/data/experiment_lists/{{exp_list}}/labelling.json'.format(ENV_DIR), exp_list=EXPERIMENT_LISTS)
    run:
        commands = expand(
            'python2.7 {project_dir}/manage_data/labelize/label_data.py {env_dir} {{exp_list}} all_cell_types'.format(
                project_dir=PROJECT_DIR, 
                env_dir=ENV_DIR 
            ),
            exp_list=EXPERIMENT_LISTS
        )
        for c in commands:
            shell(c)

rule generate_label_graphs_2:
    input:
        expand('{}/data/experiment_lists/{{exp_list}}/experiment_list.json'.format(ENV_DIR), exp_list=EXPERIMENT_LISTS_2)
    output:
        expand('{}/data/experiment_lists/{{exp_list}}/labelling.json'.format(ENV_DIR), exp_list=EXPERIMENT_LISTS_2)
    run:
        commands = expand(
            'python2.7 {project_dir}/manage_data/labelize/label_data.py {env_dir} {{exp_list}} all_cell_types'.format(
                project_dir=PROJECT_DIR,
                env_dir=ENV_DIR
            ),
            exp_list=EXPERIMENT_LISTS_2
        )
        for c in commands:
            shell(c)

rule generate_label_graphs_3:
    input:
        expand('{}/data/experiment_lists/{{exp_list}}/experiment_list.json'.format(ENV_DIR), exp_list=EXPERIMENT_LISTS_3)
    output:
        expand('{}/data/experiment_lists/{{exp_list}}/labelling.json'.format(ENV_DIR), exp_list=EXPERIMENT_LISTS_3)
    run:
        commands = expand(
            'python2.7 {project_dir}/manage_data/labelize/label_data.py {env_dir} {{exp_list}} all_cell_types'.format(
                project_dir=PROJECT_DIR,
                env_dir=ENV_DIR
            ),
            exp_list=EXPERIMENT_LISTS_3
        )
        for c in commands:
            shell(c)

rule generate_collapsed_label_graphs:
    input:
        expand('{}/data/experiment_lists/{{exp_list}}/experiment_list.json'.format(ENV_DIR), exp_list=EXPERIMENT_LISTS)
    output:
        expand('{}/data/experiment_lists/{{exp_list}}/labelling_collapsed_dag.json'.format(ENV_DIR), exp_list=EXPERIMENT_LISTS)
    run:
        commands = expand(
            'python2.7 {project_dir}/manage_data/labelize/label_data_collapsed_dag.py {env_dir} {{exp_list}} all_cell_types'.format(
                project_dir=PROJECT_DIR,
                env_dir=ENV_DIR
            ),
            exp_list=EXPERIMENT_LISTS
        )
        for c in commands:
            shell(c)

#####################################################################
#   Use the annotation to create the data set file
#####################################################################

rule extra_data_set:
    input:
        '{}/data/data_set_metadata.json'.format(ENV_DIR)
    output:
        '{}/data/extra_data_set_metadata.json'.format(ENV_DIR)
    shell:
        '{project_dir}/manage_data/create_extra_experiment_lists/create_extra_dataset.py {env_dir} -o {{output}}'.format(
            project_dir=PROJECT_DIR,
            env_dir=ENV_DIR
        ) 

rule apply_annotation:
    input:
        '{}/manage_data/annotations/from.cell_type_primary_cells_v1-4/cell_type_primary_cells_v1-4.annot_5-1.experiment_centric.json'.format(PROJECT_DIR)        
    output:
        '{}/data/data_set_metadata.json'.format(ENV_DIR) 
    shell:
        'python2.7 {}/manage_data/annotations/from.cell_type_primary_cells_v1-4/apply_annot.metasra_v1-4.annot_5-1.experiment_centric.py'.format(PROJECT_DIR) 
