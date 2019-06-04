####################################################################
#   When the annotation changes, these variables need to be updated
####################################################################
ANNOTATION_F = '/ua/mnbernstein/projects/tbcp/phenotyping/manage_data/annotations/from.cell_type_primary_cells_v1-4/cell_type_primary_cells_v1-4.annot_6.experiment_centric.json'
ENV_NAME = 'cell_type.metasra_1-4.annot_6'

ENV_DIR = '/tier2/deweylab/mnbernstein/phenotyping_environments/{}'.format(ENV_NAME)
PROJECT_DIR = '/ua/mnbernstein/projects/tbcp/phenotyping'
TMP_DIR = '/scratch/mnbernstein/cell_type_phenotyping_tmp'
EXPERIMENT_LISTS = [
    'all_untampered_bulk_primary_cells_with_data',
    'untampered_bulk_primary_cells_with_data',
    'train_set.untampered_bulk_primary_cells_with_data',
    'test_set.untampered_bulk_primary_cells_with_data',
    'untampered_single_cell_primary_cells_with_data'
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

ISOTONIC_LOGISTIC_L2_LOG_CPM_PSUEDO_FULL_DOWNWEIGHT = "ir.log_reg.l2.downweight.cpm_psuedo"


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

ANALYZE_BULK_ALGOS=[
    ISOTONIC_LOGISTIC_L2_LOG_CPM_ASSERT_AMBIG_NEG_FULL,
    ISOTONIC_LOGISTIC_L2_LOG_CPM_ASSERT_AMBIG_NEG_FULL_DOWNWEIGHT,
    TPR_LOGISTIC_L2_LOG_CPM_FULL,
    TPR_LOGISTIC_L2_LOG_CPM_FULL_DOWNWEIGHT,
    CDC_LOG_CPM_LOGISTIC_L2_ASSERT_AMBIG_NEG_FULL, 
    CDC_LOG_CPM_LOGISTIC_L2_ASSERT_AMBIG_NEG_FULL_DOWNWEIGHT, 
    BNC_LOG_CPM_LINEAR_SVM_COND_ASSERT_AMBIG_NEG, 
    PER_LABEL_LOG_CPM_LOGISTIC_L2_FULL,     
    PER_LABEL_LOG_CPM_LOGISTIC_L2_FULL_DOWNWEIGHT,     
    ONE_NN_LOG_CPM_CORRELATION     
]

ANALYZE_BULK_PLOTS = [
    'achievable_recall_boxplot.eps',  
    'avg_prec_boxplot.pdf',                 
    'paired_avg_prec_boxplot.eps',  
    'pan_data_set_pr_curves.pdf',           
    'pr_curves_on_graph.pdf',                  
    'win_diff_achievable_recall_heatmap.pdf',
    'achievable_recall_boxplot.pdf',  
    'paired_achievable_recall_boxplot.eps',  
    'paired_avg_prec_boxplot.pdf', 
    'avg_prec_boxplot.eps',           
    'paired_achievable_recall_boxplot.pdf',
    'pan_data_set_pr_curves.eps',   
    'win_diff_avg_prec_heatmap.pdf'
]

rule all:
    input:
        expand('{}/data/experiment_lists/{{exp_list}}/experiment_list.json'.format(ENV_DIR), exp_list=EXPERIMENT_LISTS),
        expand('{}/data/experiment_lists/{{exp_list}}/labelling.json'.format(ENV_DIR), exp_list=EXPERIMENT_LISTS)
        
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

rule analyze_effects_of_heterogeneous_data_analysis:
    input:
        '{}/data/experiment_lists/untampered_bulk_primary_cells_with_data/analyze_effects_of_heterogeneous_data/homo_vs_hetero_training_results.min_10.json'.format(ENV_DIR)
    output:
        '{}/data/experiment_lists/untampered_bulk_primary_cells_with_data/analyze_effects_of_heterogeneous_data/compare_avg_precs_homo_heter.pdf'
    run:
        commands=[
            'python2.7 {project_dir}/learning/analyze_effects_of_heterogeneous_data/assess_results.py {{input}} {env} untampered_bulk_primary_cells_with_data -o {env}/data/experiment_lists/untampered_bulk_primary_cells_with_data/analyze_effects_of_heterogeneous_data'.format(
                project_dir=PROJECT_DIR,
                env=ENV_DIR
            )
        ]
        for c in commands:
            shell(c)

####################################################################
#   Perform GO gene set enrichment analysis on coefficients
####################################################################

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
            'python2.7 {project_dir}/learning/leave_study_out_condorized/leave_study_out_cv_condor_primary.py {project_dir}/learning/train_config/{env_name}/{cv_config}.json {env_dir} untampered_bulk_primary_cells_with_data -o {{output}} -r {tmp_dir}/leave_study_out_cv_condor/untampered_bulk_primary_cells_with_data/{cv_config}'.format(
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

#######################################################################
#   Generate plots from training on the bulk training set and applying
#   the model to the bulk test set
#######################################################################
rule analyze_bulk_test_set:
    input:
        expand(
            '{env_dir}/data/experiment_lists/{test_exp_list}/trained_on_another_experiment_list/{train_exp_list}/{{algo_config}}/prediction_results.json'.format(
                env_dir=ENV_DIR,
                test_exp_list='test_set.untampered_bulk_primary_cells_with_data',
                train_exp_list='train_set.untampered_bulk_primary_cells_with_data'
            ), 
            algo_config=ANALYZE_BULK_ALGOS
        )
    output:
        expand(
            '{env_dir}/data/experiment_lists/{test_exp_list}/trained_on_another_experiment_list/{train_exp_list}/summary/{{plot_f}}'.format(
                env_dir=ENV_DIR,
                test_exp_list='test_set.untampered_bulk_primary_cells_with_data',
                train_exp_list='train_set.untampered_bulk_primary_cells_with_data'
            ),
            plot_f=ANALYZE_BULK_PLOTS
        )
    run:
        commands=[
            'mkdir -p {env_dir}/data/experiment_lists/{test_exp_list}/trained_on_another_experiment_list/{train_exp_list}/summary'.format(
                env_dir=ENV_DIR,
                test_exp_list='test_set.untampered_bulk_primary_cells_with_data',
                train_exp_list='train_set.untampered_bulk_primary_cells_with_data'
            ),  
            'python2.7 {project_dir}/learning/evaluation/compare_mult_methods.py {project_dir}/learning/evaluation/method_comparison_configs/bulk_test_set.json'.format(
                project_dir=PROJECT_DIR
            )
        ]
        for c in commands:
            shell(c)

        
#####################################################################
#   Algorithm: Indpendent classifiers
#   Weighted loss function: no
#   Training set: training set
#####################################################################
rule train_training_ind:
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
            'python2.7 {project_dir}/learning/train_and_save.py {project_dir}/learning/train_config/{env_name}/{algo_config}.json {env_dir} {exp_list} -o {env_dir}/data/experiment_lists/{exp_list}/trained_models/{algo_config} -r /scratch/mnbernstein/cell_type_phenotyping_tmp/train_and_save_artifacts'.format(
                env_dir=ENV_DIR,
                env_name=ENV_NAME,
                exp_list='train_set.untampered_bulk_primary_cells_with_data',
                algo_config=PER_LABEL_LOG_CPM_LOGISTIC_L2_FULL,
                project_dir=PROJECT_DIR
            )
        ]
        for c in commands:
            shell(c)

rule apply_test_ind:
    input:
        test_exp_list = '{env_dir}/data/experiment_lists/{test_exp_list}/experiment_list.json'.format(
            env_dir=ENV_DIR,
            test_exp_list='test_set.untampered_bulk_primary_cells_with_data'
        ),
        labelling_f = '{env_dir}/data/experiment_lists/{test_exp_list}/labelling.json'.format(
            env_dir=ENV_DIR,
            test_exp_list='test_set.untampered_bulk_primary_cells_with_data'
        ),
        model_f = '{env_dir}/data/experiment_lists/{train_exp_list}/trained_models/{algo_config}/model.pickle'.format(
            env_dir=ENV_DIR,
            train_exp_list='train_set.untampered_bulk_primary_cells_with_data',
            algo_config=PER_LABEL_LOG_CPM_LOGISTIC_L2_FULL
        )
    output:
        '{env_dir}/data/experiment_lists/{test_exp_list}/trained_on_another_experiment_list/{train_exp_list}/{algo_config}/prediction_results.json'.format(
            env_dir=ENV_DIR,
            test_exp_list='test_set.untampered_bulk_primary_cells_with_data',
            train_exp_list='train_set.untampered_bulk_primary_cells_with_data',
            algo_config=PER_LABEL_LOG_CPM_LOGISTIC_L2_FULL
        )
    run:
        commands = [
            'mkdir -p {env_dir}/data/experiment_lists/{test_exp_list}/trained_on_another_experiment_list/{train_exp_list}/{algo_config}'.format(
                env_dir=ENV_DIR,
                test_exp_list='test_set.untampered_bulk_primary_cells_with_data',
                train_exp_list='train_set.untampered_bulk_primary_cells_with_data',
                algo_config=PER_LABEL_LOG_CPM_LOGISTIC_L2_FULL
            ),
            'python2.7 {project_dir}/learning/apply_saved_model.py {{input.model_f}} {env_dir} {test_exp_list} -o {env_dir}/data/experiment_lists/{test_exp_list}/trained_on_another_experiment_list/{train_exp_list}/{algo_config}'.format(
                env_dir=ENV_DIR,
                project_dir=PROJECT_DIR,
                test_exp_list='test_set.untampered_bulk_primary_cells_with_data',
                train_exp_list='train_set.untampered_bulk_primary_cells_with_data',
                algo_config=PER_LABEL_LOG_CPM_LOGISTIC_L2_FULL
            )
        ]
        for c in commands:
            shell(c)




#####################################################################
#   Algorithm: Indpendent classifiers
#   Weighted loss function: yes
#   Training set: training set
#####################################################################
rule train_training_ind_downweight:
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
            'python2.7 {project_dir}/learning/train_and_save.py {project_dir}/learning/train_config/{env_name}/{algo_config}.json {env_dir} {exp_list} -o {env_dir}/data/experiment_lists/{exp_list}/trained_models/{algo_config} -r /scratch/mnbernstein/cell_type_phenotyping_tmp/train_and_save_artifacts'.format(
                env_dir=ENV_DIR,
                env_name=ENV_NAME,
                exp_list='train_set.untampered_bulk_primary_cells_with_data',
                algo_config=PER_LABEL_LOG_CPM_LOGISTIC_L2_FULL_DOWNWEIGHT,
                project_dir=PROJECT_DIR
            )
        ]
        for c in commands:
            shell(c)

rule apply_test_ind_downweight:
    input:
        test_exp_list = '{env_dir}/data/experiment_lists/{test_exp_list}/experiment_list.json'.format(
            env_dir=ENV_DIR,
            test_exp_list='test_set.untampered_bulk_primary_cells_with_data'
        ),
        labelling_f = '{env_dir}/data/experiment_lists/{test_exp_list}/labelling.json'.format(
            env_dir=ENV_DIR,
            test_exp_list='test_set.untampered_bulk_primary_cells_with_data'
        ),
        model_f = '{env_dir}/data/experiment_lists/{train_exp_list}/trained_models/{algo_config}/model.pickle'.format(
            env_dir=ENV_DIR,
            train_exp_list='train_set.untampered_bulk_primary_cells_with_data',
            algo_config=PER_LABEL_LOG_CPM_LOGISTIC_L2_FULL_DOWNWEIGHT
        )
    output:
        '{env_dir}/data/experiment_lists/{test_exp_list}/trained_on_another_experiment_list/{train_exp_list}/{algo_config}/prediction_results.json'.format(
            env_dir=ENV_DIR,
            test_exp_list='test_set.untampered_bulk_primary_cells_with_data',
            train_exp_list='train_set.untampered_bulk_primary_cells_with_data',
            algo_config=PER_LABEL_LOG_CPM_LOGISTIC_L2_FULL_DOWNWEIGHT
        )
    run:
        commands = [
            'mkdir -p {env_dir}/data/experiment_lists/{test_exp_list}/trained_on_another_experiment_list/{train_exp_list}/{algo_config}'.format(
                env_dir=ENV_DIR,
                test_exp_list='test_set.untampered_bulk_primary_cells_with_data',
                train_exp_list='train_set.untampered_bulk_primary_cells_with_data',
                algo_config=PER_LABEL_LOG_CPM_LOGISTIC_L2_FULL_DOWNWEIGHT
            ),
            'python2.7 {project_dir}/learning/apply_saved_model.py {{input.model_f}} {env_dir} {test_exp_list} -o {env_dir}/data/experiment_lists/{test_exp_list}/trained_on_another_experiment_list/{train_exp_list}/{algo_config}'.format(
                env_dir=ENV_DIR,
                project_dir=PROJECT_DIR,
                test_exp_list='test_set.untampered_bulk_primary_cells_with_data',
                train_exp_list='train_set.untampered_bulk_primary_cells_with_data',
                algo_config=PER_LABEL_LOG_CPM_LOGISTIC_L2_FULL_DOWNWEIGHT
            )
        ]
        for c in commands:
            shell(c)

#####################################################################
#   Algorithm: CLR
#   Weighted loss function: yes
#   Training set: training set
#####################################################################
rule train_training_clr_downweight:
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
            'python2.7 {project_dir}/learning/train_and_save.py {project_dir}/learning/train_config/{env_name}/{algo_config}.json {env_dir} {exp_list} -o {env_dir}/data/experiment_lists/{exp_list}/trained_models/{algo_config} -r /scratch/mnbernstein/cell_type_phenotyping_tmp/train_and_save_artifacts'.format(
                env_dir=ENV_DIR,
                env_name=ENV_NAME,
                exp_list='train_set.untampered_bulk_primary_cells_with_data',
                algo_config=CDC_LOG_CPM_LOGISTIC_L2_ASSERT_AMBIG_NEG_FULL_DOWNWEIGHT,
                project_dir=PROJECT_DIR
            )
        ]
        for c in commands:
            shell(c)


rule apply_test_clr_downweight:
    input:
        test_exp_list = '{env_dir}/data/experiment_lists/{test_exp_list}/experiment_list.json'.format(
            env_dir=ENV_DIR,
            test_exp_list='test_set.untampered_bulk_primary_cells_with_data'
        ),
        labelling_f = '{env_dir}/data/experiment_lists/{test_exp_list}/labelling.json'.format(
            env_dir=ENV_DIR,
            test_exp_list='test_set.untampered_bulk_primary_cells_with_data'
        ),
        model_f = '{env_dir}/data/experiment_lists/{train_exp_list}/trained_models/{algo_config}/model.pickle'.format(
            env_dir=ENV_DIR,
            train_exp_list='train_set.untampered_bulk_primary_cells_with_data',
            algo_config=CDC_LOG_CPM_LOGISTIC_L2_ASSERT_AMBIG_NEG_FULL_DOWNWEIGHT
        )
    output:
        '{env_dir}/data/experiment_lists/{test_exp_list}/trained_on_another_experiment_list/{train_exp_list}/{algo_config}/prediction_results.json'.format(
            env_dir=ENV_DIR,
            test_exp_list='test_set.untampered_bulk_primary_cells_with_data',
            train_exp_list='train_set.untampered_bulk_primary_cells_with_data',
            algo_config=CDC_LOG_CPM_LOGISTIC_L2_ASSERT_AMBIG_NEG_FULL_DOWNWEIGHT
        )
    run:
        commands = [
            'mkdir -p {env_dir}/data/experiment_lists/{test_exp_list}/trained_on_another_experiment_list/{train_exp_list}/{algo_config}'.format(
                env_dir=ENV_DIR,
                test_exp_list='test_set.untampered_bulk_primary_cells_with_data',
                train_exp_list='train_set.untampered_bulk_primary_cells_with_data',
                algo_config=CDC_LOG_CPM_LOGISTIC_L2_ASSERT_AMBIG_NEG_FULL_DOWNWEIGHT
            ),
            'python2.7 {project_dir}/learning/apply_saved_model.py {{input.model_f}} {env_dir} {test_exp_list} -o {env_dir}/data/experiment_lists/{test_exp_list}/trained_on_another_experiment_list/{train_exp_list}/{algo_config}'.format(
                env_dir=ENV_DIR,
                project_dir=PROJECT_DIR,
                test_exp_list='test_set.untampered_bulk_primary_cells_with_data',
                train_exp_list='train_set.untampered_bulk_primary_cells_with_data',
                algo_config=CDC_LOG_CPM_LOGISTIC_L2_ASSERT_AMBIG_NEG_FULL_DOWNWEIGHT
            )
        ]
        for c in commands:
            shell(c)


#####################################################################
#   Algorithm: CLR
#   Weighted loss function: yes
#   Training set: all samples
#####################################################################
rule train_all_clr_downweight:
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
            'python2.7 {project_dir}/learning/train_and_save.py {project_dir}/learning/train_config/{env_name}/{algo_config}.json {env_dir} {exp_list} -o {env_dir}/data/experiment_lists/{exp_list}/trained_models/{algo_config} -r /scratch/mnbernstein/cell_type_phenotyping_tmp/train_and_save_artifacts'.format(
                env_dir=ENV_DIR,
                env_name=ENV_NAME,
                exp_list='untampered_bulk_primary_cells_with_data',
                algo_config=CDC_LOG_CPM_LOGISTIC_L2_ASSERT_AMBIG_NEG_FULL_DOWNWEIGHT,
                project_dir=PROJECT_DIR
            )
        ]
        for c in commands:
            shell(c)

rule apply_sc_clr_logistic_downweight:
    input:
        test_exp_list = '{env_dir}/data/experiment_lists/{test_exp_list}/experiment_list.json'.format(
            env_dir=ENV_DIR,
            test_exp_list='untampered_single_cell_primary_cells_with_data_cell_types_in_bulk'
        ),
        labelling_f = '{env_dir}/data/experiment_lists/{test_exp_list}/labelling.json'.format(
            env_dir=ENV_DIR,
            test_exp_list='untampered_single_cell_primary_cells_with_data_cell_types_in_bulk'
        ),
        model_f = '{env_dir}/data/experiment_lists/{train_exp_list}/trained_models/{algo_config}/model.pickle'.format(
            env_dir=ENV_DIR,
            train_exp_list='untampered_bulk_primary_cells_with_data',
            algo_config=CDC_LOG_CPM_LOGISTIC_L2_ASSERT_AMBIG_NEG_FULL_DOWNWEIGHT
        )
    output:
        '{env_dir}/data/experiment_lists/{test_exp_list}/trained_on_another_experiment_list/{train_exp_list}/{algo_config}/prediction_results.json'.format(
            env_dir=ENV_DIR,
            test_exp_list='untampered_single_cell_primary_cells_with_data_cell_types_in_bulk',
            train_exp_list='untampered_bulk_primary_cells_with_data',
            algo_config=CDC_LOG_CPM_LOGISTIC_L2_ASSERT_AMBIG_NEG_FULL_DOWNWEIGHT
        )
    run:
        commands = [
            'mkdir -p {env_dir}/data/experiment_lists/{test_exp_list}/trained_on_another_experiment_list/{train_exp_list}/{algo_config}'.format(
                env_dir=ENV_DIR,
                test_exp_list='untampered_single_cell_primary_cells_with_data_cell_types_in_bulk',
                train_exp_list='untampered_bulk_primary_cells_with_data',
                algo_config=CDC_LOG_CPM_LOGISTIC_L2_ASSERT_AMBIG_NEG_FULL_DOWNWEIGHT
            ),
            'python2.7 {project_dir}/learning/apply_saved_model.py {{input.model_f}} {env_dir} {test_exp_list} -o {env_dir}/data/experiment_lists/{test_exp_list}/trained_on_another_experiment_list/{train_exp_list}/{algo_config}'.format(
                env_dir=ENV_DIR,
                project_dir=PROJECT_DIR,
                test_exp_list='untampered_single_cell_primary_cells_with_data_cell_types_in_bulk',
                train_exp_list='untampered_bulk_primary_cells_with_data',
                algo_config=CDC_LOG_CPM_LOGISTIC_L2_ASSERT_AMBIG_NEG_FULL_DOWNWEIGHT
            )
        ]
        for c in commands:
            shell(c)

#####################################################################
#   Algorithm: CLR
#   Weighted loss function: no
#   Trained on: training set
#####################################################################
rule train_training_clr:
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
            'python2.7 {project_dir}/learning/train_and_save.py {project_dir}/learning/train_config/{env_name}/{algo_config}.json {env_dir} {exp_list} -o {env_dir}/data/experiment_lists/{exp_list}/trained_models/{algo_config} -r /scratch/mnbernstein/cell_type_phenotyping_tmp/train_and_save_artifacts'.format(
                env_dir=ENV_DIR,
                env_name=ENV_NAME,
                exp_list='train_set.untampered_bulk_primary_cells_with_data',
                algo_config=CDC_LOG_CPM_LOGISTIC_L2_ASSERT_AMBIG_NEG_FULL,
                project_dir=PROJECT_DIR
            )
        ]
        for c in commands:
            shell(c)    

rule apply_test_clr:
    input:
        test_exp_list = '{env_dir}/data/experiment_lists/{test_exp_list}/experiment_list.json'.format(
            env_dir=ENV_DIR,
            test_exp_list='test_set.untampered_bulk_primary_cells_with_data'
        ),
        labelling_f = '{env_dir}/data/experiment_lists/{test_exp_list}/labelling.json'.format(
            env_dir=ENV_DIR,
            test_exp_list='test_set.untampered_bulk_primary_cells_with_data'
        ),
        model_f = '{env_dir}/data/experiment_lists/{train_exp_list}/trained_models/{algo_config}/model.pickle'.format(
            env_dir=ENV_DIR,
            train_exp_list='train_set.untampered_bulk_primary_cells_with_data',
            algo_config=CDC_LOG_CPM_LOGISTIC_L2_ASSERT_AMBIG_NEG_FULL
        )
    output:
        '{env_dir}/data/experiment_lists/{test_exp_list}/trained_on_another_experiment_list/{train_exp_list}/{algo_config}/prediction_results.json'.format(
            env_dir=ENV_DIR,
            test_exp_list='test_set.untampered_bulk_primary_cells_with_data',
            train_exp_list='train_set.untampered_bulk_primary_cells_with_data',
            algo_config=CDC_LOG_CPM_LOGISTIC_L2_ASSERT_AMBIG_NEG_FULL
        )
    run:
        commands = [
            'mkdir -p {env_dir}/data/experiment_lists/{test_exp_list}/trained_on_another_experiment_list/{train_exp_list}/{algo_config}'.format(
                env_dir=ENV_DIR,
                test_exp_list='test_set.untampered_bulk_primary_cells_with_data',
                train_exp_list='train_set.untampered_bulk_primary_cells_with_data',
                algo_config=CDC_LOG_CPM_LOGISTIC_L2_ASSERT_AMBIG_NEG_FULL
            ),
            'python2.7 {project_dir}/learning/apply_saved_model.py {{input.model_f}} {env_dir} {test_exp_list} -o {env_dir}/data/experiment_lists/{test_exp_list}/trained_on_another_experiment_list/{train_exp_list}/{algo_config}'.format(
                env_dir=ENV_DIR,
                project_dir=PROJECT_DIR,
                test_exp_list='test_set.untampered_bulk_primary_cells_with_data',
                train_exp_list='train_set.untampered_bulk_primary_cells_with_data',
                algo_config=CDC_LOG_CPM_LOGISTIC_L2_ASSERT_AMBIG_NEG_FULL
            )
        ]
        for c in commands:
            shell(c)


#####################################################################
#   Algorithm: IR
#   Weighted loss function: no
#   Trained on: training set
#####################################################################
rule train_training_ir:
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
            'python2.7 {project_dir}/learning/train_and_save.py {project_dir}/learning/train_config/{env_name}/{algo_config}.json {env_dir} {exp_list} -o {env_dir}/data/experiment_lists/{exp_list}/trained_models/{algo_config} -r /scratch/mnbernstein/cell_type_phenotyping_tmp/train_and_save_artifacts'.format(
                env_dir=ENV_DIR,
                env_name=ENV_NAME,
                exp_list='train_set.untampered_bulk_primary_cells_with_data',
                algo_config=ISOTONIC_LOGISTIC_L2_LOG_CPM_ASSERT_AMBIG_NEG_FULL,
                project_dir=PROJECT_DIR
            )
        ]
        for c in commands:
            shell(c)

rule apply_test_ir:
    input:
        test_exp_list = '{env_dir}/data/experiment_lists/{test_exp_list}/experiment_list.json'.format(
            env_dir=ENV_DIR,
            test_exp_list='test_set.untampered_bulk_primary_cells_with_data'
        ),
        labelling_f = '{env_dir}/data/experiment_lists/{test_exp_list}/labelling.json'.format(
            env_dir=ENV_DIR,
            test_exp_list='test_set.untampered_bulk_primary_cells_with_data'
        ),
        model_f = '{env_dir}/data/experiment_lists/{train_exp_list}/trained_models/{algo_config}/model.pickle'.format(
            env_dir=ENV_DIR,
            train_exp_list='train_set.untampered_bulk_primary_cells_with_data',
            algo_config=ISOTONIC_LOGISTIC_L2_LOG_CPM_ASSERT_AMBIG_NEG_FULL
        )
    output:
        '{env_dir}/data/experiment_lists/{test_exp_list}/trained_on_another_experiment_list/{train_exp_list}/{algo_config}/prediction_results.json'.format(
            env_dir=ENV_DIR,
            test_exp_list='test_set.untampered_bulk_primary_cells_with_data',
            train_exp_list='train_set.untampered_bulk_primary_cells_with_data',
            algo_config=ISOTONIC_LOGISTIC_L2_LOG_CPM_ASSERT_AMBIG_NEG_FULL
        )
    run:
        commands = [
            'mkdir -p {env_dir}/data/experiment_lists/{test_exp_list}/trained_on_another_experiment_list/{train_exp_list}/{algo_config}'.format(
                env_dir=ENV_DIR,
                test_exp_list='test_set.untampered_bulk_primary_cells_with_data',
                train_exp_list='train_set.untampered_bulk_primary_cells_with_data',
                algo_config=ISOTONIC_LOGISTIC_L2_LOG_CPM_ASSERT_AMBIG_NEG_FULL
            ),
            'python2.7 {project_dir}/learning/apply_saved_model.py {{input.model_f}} {env_dir} {test_exp_list} -o {env_dir}/data/experiment_lists/{test_exp_list}/trained_on_another_experiment_list/{train_exp_list}/{algo_config}'.format(
                env_dir=ENV_DIR,
                project_dir=PROJECT_DIR,
                test_exp_list='test_set.untampered_bulk_primary_cells_with_data',
                train_exp_list='train_set.untampered_bulk_primary_cells_with_data',
                algo_config=ISOTONIC_LOGISTIC_L2_LOG_CPM_ASSERT_AMBIG_NEG_FULL
            )
        ]
        for c in commands:
            shell(c)


#####################################################################
#   Algorithm: IR
#   Weighted loss function: yes
#   Trained on: training set
#####################################################################
rule train_training_ir_downweight:
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
            'python2.7 {project_dir}/learning/train_and_save.py {project_dir}/learning/train_config/{env_name}/{algo_config}.json {env_dir} {exp_list} -o {env_dir}/data/experiment_lists/{exp_list}/trained_models/{algo_config} -r /scratch/mnbernstein/cell_type_phenotyping_tmp/train_and_save_artifacts'.format(
                env_dir=ENV_DIR,
                env_name=ENV_NAME,
                exp_list='train_set.untampered_bulk_primary_cells_with_data',
                algo_config=ISOTONIC_LOGISTIC_L2_LOG_CPM_ASSERT_AMBIG_NEG_FULL_DOWNWEIGHT,
                project_dir=PROJECT_DIR
            )
        ]
        for c in commands:
            shell(c)

rule apply_test_ir_downweight:
    input:
        test_exp_list = '{env_dir}/data/experiment_lists/{test_exp_list}/experiment_list.json'.format(
            env_dir=ENV_DIR,
            test_exp_list='test_set.untampered_bulk_primary_cells_with_data'
        ),
        labelling_f = '{env_dir}/data/experiment_lists/{test_exp_list}/labelling.json'.format(
            env_dir=ENV_DIR,
            test_exp_list='test_set.untampered_bulk_primary_cells_with_data'
        ),
        model_f = '{env_dir}/data/experiment_lists/{train_exp_list}/trained_models/{algo_config}/model.pickle'.format(
            env_dir=ENV_DIR,
            train_exp_list='train_set.untampered_bulk_primary_cells_with_data',
            algo_config=ISOTONIC_LOGISTIC_L2_LOG_CPM_ASSERT_AMBIG_NEG_FULL_DOWNWEIGHT
        )
    output:
        '{env_dir}/data/experiment_lists/{test_exp_list}/trained_on_another_experiment_list/{train_exp_list}/{algo_config}/prediction_results.json'.format(
            env_dir=ENV_DIR,
            test_exp_list='test_set.untampered_bulk_primary_cells_with_data',
            train_exp_list='train_set.untampered_bulk_primary_cells_with_data',
            algo_config=ISOTONIC_LOGISTIC_L2_LOG_CPM_ASSERT_AMBIG_NEG_FULL_DOWNWEIGHT
        )
    run:
        commands = [
            'mkdir -p {env_dir}/data/experiment_lists/{test_exp_list}/trained_on_another_experiment_list/{train_exp_list}/{algo_config}'.format(
                env_dir=ENV_DIR,
                test_exp_list='test_set.untampered_bulk_primary_cells_with_data',
                train_exp_list='train_set.untampered_bulk_primary_cells_with_data',
                algo_config=ISOTONIC_LOGISTIC_L2_LOG_CPM_ASSERT_AMBIG_NEG_FULL_DOWNWEIGHT
            ),
            'python2.7 {project_dir}/learning/apply_saved_model.py {{input.model_f}} {env_dir} {test_exp_list} -o {env_dir}/data/experiment_lists/{test_exp_list}/trained_on_another_experiment_list/{train_exp_list}/{algo_config}'.format(
                env_dir=ENV_DIR,
                project_dir=PROJECT_DIR,
                test_exp_list='test_set.untampered_bulk_primary_cells_with_data',
                train_exp_list='train_set.untampered_bulk_primary_cells_with_data',
                algo_config=ISOTONIC_LOGISTIC_L2_LOG_CPM_ASSERT_AMBIG_NEG_FULL_DOWNWEIGHT
            )
        ]
        for c in commands:
            shell(c)


#####################################################################
#   Algorithm: IR
#   Weighted loss function: yes
#   Trained on: all samples
#####################################################################
rule train_all_ir_downweight:
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
            'python2.7 {project_dir}/learning/train_and_save.py {project_dir}/learning/train_config/{env_name}/{algo_config}.json {env_dir} {exp_list} -o {env_dir}/data/experiment_lists/{exp_list}/trained_models/{algo_config} -r /scratch/mnbernstein/cell_type_phenotyping_tmp/train_and_save_artifacts'.format(
                env_dir=ENV_DIR,
                env_name=ENV_NAME,
                exp_list='untampered_bulk_primary_cells_with_data',
                algo_config=ISOTONIC_LOGISTIC_L2_LOG_CPM_ASSERT_AMBIG_NEG_FULL_DOWNWEIGHT,
                project_dir=PROJECT_DIR
            )
        ]
        for c in commands:
            shell(c)

rule apply_sc_ir_downweight:
    input:
        test_exp_list = '{env_dir}/data/experiment_lists/{test_exp_list}/experiment_list.json'.format(
            env_dir=ENV_DIR,
            test_exp_list='untampered_single_cell_primary_cells_with_data_cell_types_in_bulk'
        ),
        labelling_f = '{env_dir}/data/experiment_lists/{test_exp_list}/labelling.json'.format(
            env_dir=ENV_DIR,
            test_exp_list='untampered_single_cell_primary_cells_with_data_cell_types_in_bulk'
        ),
        model_f = '{env_dir}/data/experiment_lists/{train_exp_list}/trained_models/{algo_config}/model.pickle'.format(
            env_dir=ENV_DIR,
            train_exp_list='untampered_bulk_primary_cells_with_data',
            algo_config=ISOTONIC_LOGISTIC_L2_LOG_CPM_ASSERT_AMBIG_NEG_FULL_DOWNWEIGHT
        )
    output:
        '{env_dir}/data/experiment_lists/{test_exp_list}/trained_on_another_experiment_list/{train_exp_list}/{algo_config}/prediction_results.json'.format(
            env_dir=ENV_DIR,
            test_exp_list='untampered_single_cell_primary_cells_with_data_cell_types_in_bulk',
            train_exp_list='untampered_bulk_primary_cells_with_data',
            algo_config=ISOTONIC_LOGISTIC_L2_LOG_CPM_ASSERT_AMBIG_NEG_FULL_DOWNWEIGHT
        )
    run:
        commands = [
            'mkdir -p {env_dir}/data/experiment_lists/{test_exp_list}/trained_on_another_experiment_list/{train_exp_list}/{algo_config}'.format(
                env_dir=ENV_DIR,
                test_exp_list='untampered_single_cell_primary_cells_with_data_cell_types_in_bulk',
                train_exp_list='untampered_bulk_primary_cells_with_data',
                algo_config=ISOTONIC_LOGISTIC_L2_LOG_CPM_ASSERT_AMBIG_NEG_FULL_DOWNWEIGHT
            ),
            'python2.7 {project_dir}/learning/apply_saved_model.py {{input.model_f}} {env_dir} {test_exp_list} -o {env_dir}/data/experiment_lists/{test_exp_list}/trained_on_another_experiment_list/{train_exp_list}/{algo_config}'.format(
                env_dir=ENV_DIR,
                project_dir=PROJECT_DIR,
                test_exp_list='untampered_single_cell_primary_cells_with_data_cell_types_in_bulk',
                train_exp_list='untampered_bulk_primary_cells_with_data',
                algo_config=ISOTONIC_LOGISTIC_L2_LOG_CPM_ASSERT_AMBIG_NEG_FULL_DOWNWEIGHT
            )
        ]
        for c in commands:
            shell(c)


#####################################################################
#   Algorithm: BNC-SVM-bins 
#   Trained on: training set
#####################################################################
rule train_training_bnc_svm:
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
            'python2.7 {project_dir}/learning/train_and_save.py {project_dir}/learning/train_config/{env_name}/{algo_config}.json {env_dir} {exp_list} -o {env_dir}/data/experiment_lists/{exp_list}/trained_models/{algo_config} -r /scratch/mnbernstein/cell_type_phenotyping_tmp/train_and_save_artifacts'.format(
                env_dir=ENV_DIR,
                env_name=ENV_NAME,
                exp_list='train_set.untampered_bulk_primary_cells_with_data',
                algo_config=BNC_LOG_CPM_LINEAR_SVM_COND_ASSERT_AMBIG_NEG,
                project_dir=PROJECT_DIR
            )
        ]
        for c in commands:
            shell(c)

rule apply_test_bnc_svm:
    input:
        test_exp_list = '{env_dir}/data/experiment_lists/{test_exp_list}/experiment_list.json'.format(
            env_dir=ENV_DIR,
            test_exp_list='test_set.untampered_bulk_primary_cells_with_data'
        ),
        labelling_f = '{env_dir}/data/experiment_lists/{test_exp_list}/labelling.json'.format(
            env_dir=ENV_DIR,
            test_exp_list='test_set.untampered_bulk_primary_cells_with_data'
        ),
        model_f = '{env_dir}/data/experiment_lists/{train_exp_list}/trained_models/{algo_config}/model.pickle'.format(
            env_dir=ENV_DIR,
            train_exp_list='train_set.untampered_bulk_primary_cells_with_data',
            algo_config=BNC_LOG_CPM_LINEAR_SVM_COND_ASSERT_AMBIG_NEG
        )
    output:
        '{env_dir}/data/experiment_lists/{test_exp_list}/trained_on_another_experiment_list/{train_exp_list}/{algo_config}/prediction_results.json'.format(
            env_dir=ENV_DIR,
            test_exp_list='test_set.untampered_bulk_primary_cells_with_data',
            train_exp_list='train_set.untampered_bulk_primary_cells_with_data',
            algo_config=BNC_LOG_CPM_LINEAR_SVM_COND_ASSERT_AMBIG_NEG
        )
    run:
        commands = [
            'mkdir -p {env_dir}/data/experiment_lists/{test_exp_list}/trained_on_another_experiment_list/{train_exp_list}/{algo_config}'.format(
                env_dir=ENV_DIR,
                test_exp_list='test_set.untampered_bulk_primary_cells_with_data',
                train_exp_list='train_set.untampered_bulk_primary_cells_with_data',
                algo_config=BNC_LOG_CPM_LINEAR_SVM_COND_ASSERT_AMBIG_NEG
            ),
            'python2.7 {project_dir}/learning/apply_saved_model_condorized/apply_saved_model_condor_primary.py {{input.model_f}} {env_dir} {test_exp_list} 5 -o {{output}} -c /scratch/mnbernstein/apply_trained_models_condor/{test_exp_list}.{algo_config}'.format(
                env_dir=ENV_DIR,
                project_dir=PROJECT_DIR,
                test_exp_list='test_set.untampered_bulk_primary_cells_with_data',
                train_exp_list='train_set.untampered_bulk_primary_cells_with_data',
                algo_config=BNC_LOG_CPM_LINEAR_SVM_COND_ASSERT_AMBIG_NEG
            )
        ]
        for c in commands:
            shell(c)

#####################################################################
#   Algorithm: NB-SVM-bins
#   Trained on: training set
#####################################################################
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
            'python2.7 {project_dir}/learning/train_and_save.py {project_dir}/learning/train_config/{env_name}/{algo_config}.json {env_dir} {exp_list} -o {env_dir}/data/experiment_lists/{exp_list}/trained_models/{algo_config} -r /scratch/mnbernstein/cell_type_phenotyping_tmp/train_and_save_artifacts'.format(
                env_dir=ENV_DIR,
                env_name=ENV_NAME,
                exp_list='train_set.untampered_bulk_primary_cells_with_data',
                algo_config=NAIVE_BAYES_SVM_COND_ASSERT_AMBIG_NEG,
                project_dir=PROJECT_DIR
            )
        ]
        for c in commands:
            shell(c)

#####################################################################
#   Alogirithm: BNC-LR-bins
#   Trained on: training set
#####################################################################
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
            'python2.7 {project_dir}/learning/train_and_save.py {project_dir}/learning/train_config/{env_name}/{algo_config}.json {env_dir} {exp_list} -o {env_dir}/data/experiment_lists/{exp_list}/trained_models/{algo_config} -r /scratch/mnbernstein/cell_type_phenotyping_tmp/train_and_save_artifacts'.format(
                env_dir=ENV_DIR,
                env_name=ENV_NAME,
                exp_list='train_set.untampered_bulk_primary_cells_with_data',
                algo_config=BNC_LOG_CPM_LOGISTIC_L2_COND,
                project_dir=PROJECT_DIR
            )
        ]
        for c in commands:
            shell(c)

#####################################################################
#   Algorithm: BNC-SVM-normal
#   Trained on: training set
#####################################################################
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
            'python2.7 {project_dir}/learning/train_and_save.py {project_dir}/learning/train_config/{env_name}/{algo_config}.json {env_dir} {exp_list} -o {env_dir}/data/experiment_lists/{exp_list}/trained_models/{algo_config} -r /scratch/mnbernstein/cell_type_phenotyping_tmp/train_and_save_artifacts'.format(
                env_dir=ENV_DIR,
                env_name=ENV_NAME,
                exp_list='train_set.untampered_bulk_primary_cells_with_data',
                algo_config=BNC_LOG_CPM_LINEAR_SVM_NORMAL,
                project_dir=PROJECT_DIR
            )
        ]
        for c in commands:
            shell(c)

#####################################################################
#  Alogirithm: TPR
#  Weighted loss function: no
#  Trained on: training set
#####################################################################
rule train_training_tpr:
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
            'python2.7 {project_dir}/learning/train_and_save.py {project_dir}/learning/train_config/{env_name}/{algo_config}.json {env_dir} {exp_list} -o {env_dir}/data/experiment_lists/{exp_list}/trained_models/{algo_config} -r /scratch/mnbernstein/cell_type_phenotyping_tmp/train_and_save_artifacts'.format(
                env_dir=ENV_DIR,
                env_name=ENV_NAME,
                exp_list='train_set.untampered_bulk_primary_cells_with_data',
                algo_config=TPR_LOGISTIC_L2_LOG_CPM_FULL,
                project_dir=PROJECT_DIR
            )
        ]
        for c in commands:
            shell(c)

rule apply_test_tpr:
    input:
        test_exp_list = '{env_dir}/data/experiment_lists/{test_exp_list}/experiment_list.json'.format(
            env_dir=ENV_DIR,
            test_exp_list='test_set.untampered_bulk_primary_cells_with_data'
        ),
        labelling_f = '{env_dir}/data/experiment_lists/{test_exp_list}/labelling.json'.format(
            env_dir=ENV_DIR,
            test_exp_list='test_set.untampered_bulk_primary_cells_with_data'
        ),
        model_f = '{env_dir}/data/experiment_lists/{train_exp_list}/trained_models/{algo_config}/model.pickle'.format(
            env_dir=ENV_DIR,
            train_exp_list='train_set.untampered_bulk_primary_cells_with_data',
            algo_config=TPR_LOGISTIC_L2_LOG_CPM_FULL
        )
    output:
        '{env_dir}/data/experiment_lists/{test_exp_list}/trained_on_another_experiment_list/{train_exp_list}/{algo_config}/prediction_results.json'.format(
            env_dir=ENV_DIR,
            test_exp_list='test_set.untampered_bulk_primary_cells_with_data',
            train_exp_list='train_set.untampered_bulk_primary_cells_with_data',
            algo_config=TPR_LOGISTIC_L2_LOG_CPM_FULL
        )
    run:
        commands = [
            'mkdir -p {env_dir}/data/experiment_lists/{test_exp_list}/trained_on_another_experiment_list/{train_exp_list}/{algo_config}'.format(
                env_dir=ENV_DIR,
                test_exp_list='test_set.untampered_bulk_primary_cells_with_data',
                train_exp_list='train_set.untampered_bulk_primary_cells_with_data',
                algo_config=TPR_LOGISTIC_L2_LOG_CPM_FULL
            ),
            'python2.7 {project_dir}/learning/apply_saved_model.py {{input.model_f}} {env_dir} {test_exp_list} -o {env_dir}/data/experiment_lists/{test_exp_list}/trained_on_another_experiment_list/{train_exp_list}/{algo_config}'.format(
                env_dir=ENV_DIR,
                project_dir=PROJECT_DIR,
                test_exp_list='test_set.untampered_bulk_primary_cells_with_data',
                train_exp_list='train_set.untampered_bulk_primary_cells_with_data',
                algo_config=TPR_LOGISTIC_L2_LOG_CPM_FULL
            )
        ]
        for c in commands:
            shell(c)

#####################################################################
#  Alogirithm: TPR
#  Weighted loss function: yes
#  Trained on: training set
#####################################################################
rule train_training_tpr_downweight:
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
            'python2.7 {project_dir}/learning/train_and_save.py {project_dir}/learning/train_config/{env_name}/{algo_config}.json {env_dir} {exp_list} -o {env_dir}/data/experiment_lists/{exp_list}/trained_models/{algo_config} -r /scratch/mnbernstein/cell_type_phenotyping_tmp/train_and_save_artifacts'.format(
                env_dir=ENV_DIR,
                env_name=ENV_NAME,
                exp_list='train_set.untampered_bulk_primary_cells_with_data',
                algo_config=TPR_LOGISTIC_L2_LOG_CPM_FULL_DOWNWEIGHT,
                project_dir=PROJECT_DIR
            )
        ]
        for c in commands:
            shell(c)

rule apply_test_tpr_downweight:
    input:
        test_exp_list = '{env_dir}/data/experiment_lists/{test_exp_list}/experiment_list.json'.format(
            env_dir=ENV_DIR,
            test_exp_list='test_set.untampered_bulk_primary_cells_with_data'
        ),
        labelling_f = '{env_dir}/data/experiment_lists/{test_exp_list}/labelling.json'.format(
            env_dir=ENV_DIR,
            test_exp_list='test_set.untampered_bulk_primary_cells_with_data'
        ),
        model_f = '{env_dir}/data/experiment_lists/{train_exp_list}/trained_models/{algo_config}/model.pickle'.format(
            env_dir=ENV_DIR,
            train_exp_list='train_set.untampered_bulk_primary_cells_with_data',
            algo_config=TPR_LOGISTIC_L2_LOG_CPM_FULL_DOWNWEIGHT
        )
    output:
        '{env_dir}/data/experiment_lists/{test_exp_list}/trained_on_another_experiment_list/{train_exp_list}/{algo_config}/prediction_results.json'.format(
            env_dir=ENV_DIR,
            test_exp_list='test_set.untampered_bulk_primary_cells_with_data',
            train_exp_list='train_set.untampered_bulk_primary_cells_with_data',
            algo_config=TPR_LOGISTIC_L2_LOG_CPM_FULL_DOWNWEIGHT
        )
    run:
        commands = [
            'mkdir -p {env_dir}/data/experiment_lists/{test_exp_list}/trained_on_another_experiment_list/{train_exp_list}/{algo_config}'.format(
                env_dir=ENV_DIR,
                test_exp_list='test_set.untampered_bulk_primary_cells_with_data',
                train_exp_list='train_set.untampered_bulk_primary_cells_with_data',
                algo_config=TPR_LOGISTIC_L2_LOG_CPM_FULL_DOWNWEIGHT
            ),
            'python2.7 {project_dir}/learning/apply_saved_model.py {{input.model_f}} {env_dir} {test_exp_list} -o {env_dir}/data/experiment_lists/{test_exp_list}/trained_on_another_experiment_list/{train_exp_list}/{algo_config}'.format(
                env_dir=ENV_DIR,
                project_dir=PROJECT_DIR,
                test_exp_list='test_set.untampered_bulk_primary_cells_with_data',
                train_exp_list='train_set.untampered_bulk_primary_cells_with_data',
                algo_config=TPR_LOGISTIC_L2_LOG_CPM_FULL_DOWNWEIGHT
            )
        ]
        for c in commands:
            shell(c)


#####################################################################
#  Alogirithm: TPR
#  Weighted loss function: yes
#  Trained on: all samples
#####################################################################
rule train_all_tpr_downweight:
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
            'python2.7 {project_dir}/learning/train_and_save.py {project_dir}/learning/train_config/{env_name}/{algo_config}.json {env_dir} {exp_list} -o {env_dir}/data/experiment_lists/{exp_list}/trained_models/{algo_config} -r /scratch/mnbernstein/cell_type_phenotyping_tmp/train_and_save_artifacts'.format(
                env_dir=ENV_DIR,
                env_name=ENV_NAME,
                exp_list='untampered_bulk_primary_cells_with_data',
                algo_config=TPR_LOGISTIC_L2_LOG_CPM_FULL_DOWNWEIGHT,
                project_dir=PROJECT_DIR
            )
        ]
        for c in commands:
            shell(c)

rule apply_sc_tpr_downweight:
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



#####################################################################
#  Alogirithm: 1NN
#  Trained on: training_set
#####################################################################
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
        '{env_dir}/data/experiment_lists/{test_exp_list}/trained_on_another_experiment_list/{train_exp_list}/{algo_config}/prediction_results.json'.format(
            env_dir=ENV_DIR,
            test_exp_list='test_set.untampered_bulk_primary_cells_with_data',
            train_exp_list='train_set.untampered_bulk_primary_cells_with_data',
            algo_config=ONE_NN_LOG_CPM_CORRELATION
        )
    run:
        commands = [
            'mkdir -p /scratch/mnbernstein/cell_type_phenotyping_tmp/train_and_save_artifacts',
            'mkdir -p {env_dir}/data/experiment_lists/{test_exp_list}/trained_on_another_experiment_list/{train_exp_list}/{algo_config}'.format(
                env_dir=ENV_DIR,
                test_exp_list='test_set.untampered_bulk_primary_cells_with_data',
                train_exp_list='train_set.untampered_bulk_primary_cells_with_data',
                algo_config=ONE_NN_LOG_CPM_CORRELATION
            ),
            'python2.7 {project_dir}/learning/train_and_apply.py {project_dir}/learning/train_config/{env_name}/{algo_config}.json {env_dir} {train_exp_list} {test_exp_list} -r /scratch/mnbernstein/cell_type_phenotyping_tmp/train_and_save_artifacts -o {env_dir}/data/experiment_lists/{test_exp_list}/trained_on_another_experiment_list/{train_exp_list}/{algo_config}'.format(
                env_dir=ENV_DIR,
                env_name=ENV_NAME,
                project_dir=PROJECT_DIR,
                test_exp_list='test_set.untampered_bulk_primary_cells_with_data',
                train_exp_list='train_set.untampered_bulk_primary_cells_with_data',
                algo_config=ONE_NN_LOG_CPM_CORRELATION
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


rule apply_trained_model_isotonic_logistic_regression_l2_assert_ambig_downweight_psuedo:
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
            algo_config=ISOTONIC_LOGISTIC_L2_LOG_CPM_PSUEDO_FULL_DOWNWEIGHT
        )
    output:
        '{env_dir}/data/experiment_lists/{exp_list}/trained_on_another_experiment_list/{train_exp_list}/{algo_config}/prediction_results.json'.format(
            env_dir=ENV_DIR,
            exp_list='test_set.untampered_bulk_primary_cells_with_data',
            train_exp_list='train_set.untampered_bulk_primary_cells_with_data',
            algo_config=ISOTONIC_LOGISTIC_L2_LOG_CPM_PSUEDO_FULL_DOWNWEIGHT
        )
    run:
        commands = [
            'mkdir -p {env_dir}/data/experiment_lists/{exp_list}/trained_on_another_experiment_list/{train_exp_list}/{algo_config}'.format(
                env_dir=ENV_DIR,
                exp_list='test_set.untampered_bulk_primary_cells_with_data',
                train_exp_list='train_set.untampered_bulk_primary_cells_with_data',
                algo_config=ISOTONIC_LOGISTIC_L2_LOG_CPM_PSUEDO_FULL_DOWNWEIGHT
            ),
            'python2.7 {project_dir}/learning/apply_saved_model.py {{input.model_f}} {env_dir} {exp_list} -o {env_dir}/data/experiment_lists/{exp_list}/trained_on_another_experiment_list/{train_exp_list}/{algo_config}'.format(
                env_dir=ENV_DIR,
                project_dir=PROJECT_DIR,
                exp_list='test_set.untampered_bulk_primary_cells_with_data',
                train_exp_list='train_set.untampered_bulk_primary_cells_with_data',
                algo_config=ISOTONIC_LOGISTIC_L2_LOG_CPM_PSUEDO_FULL_DOWNWEIGHT
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


#rule apply_trained_model_bnc_linear_svm_count_cond_prior_assert_ambig_neg_condor:
#    input:
#        test_exp_list = '{env_dir}/data/experiment_lists/{exp_list}/experiment_list.json'.format(
#            env_dir=ENV_DIR,
#            exp_list='test_set.untampered_bulk_primary_cells_with_data'
#        ),
#        labelling_f = '{env_dir}/data/experiment_lists/{exp_list}/labelling.json'.format(
#            env_dir=ENV_DIR,
#            exp_list='test_set.untampered_bulk_primary_cells_with_data'
#        ),
#        model_f = '{env_dir}/data/experiment_lists/{train_exp_list}/trained_models/{algo_config}/model.pickle'.format(
#            env_dir=ENV_DIR,
#            train_exp_list='train_set.untampered_bulk_primary_cells_with_data',
#            algo_config=BNC_LOG_CPM_LINEAR_SVM_COND_ASSERT_AMBIG_NEG
#        )
#    output:
#        '{env_dir}/data/experiment_lists/{exp_list}/trained_on_another_experiment_list/{train_exp_list}/{algo_config}/prediction_results.json'.format(
#            env_dir=ENV_DIR,
#            exp_list='test_set.untampered_bulk_primary_cells_with_data',
#            train_exp_list='train_set.untampered_bulk_primary_cells_with_data',
#            algo_config=BNC_LOG_CPM_LINEAR_SVM_COND_ASSERT_AMBIG_NEG
#        )
#    run:
#        commands = [
#            'mkdir -p {env_dir}/data/experiment_lists/{exp_list}/trained_on_another_experiment_list/{train_exp_list}/{algo_config}'.format(
#                env_dir=ENV_DIR,
#                exp_list='test_set.untampered_bulk_primary_cells_with_data',
#                train_exp_list='train_set.untampered_bulk_primary_cells_with_data',
#                algo_config=BNC_LOG_CPM_LINEAR_SVM_COND_ASSERT_AMBIG_NEG
#            ),
#            'python2.7 {project_dir}/learning/apply_saved_model_condorized/apply_saved_model_condor_primary.py {{input.model_f}} {env_dir} {exp_list} 5 -o {{output}} -c /scratch/mnbernstein/apply_trained_models_condor/{exp_list}.{algo_config}'.format(
#                env_dir=ENV_DIR,
#                project_dir=PROJECT_DIR,
#                exp_list='test_set.untampered_bulk_primary_cells_with_data',
#                train_exp_list='train_set.untampered_bulk_primary_cells_with_data',
#                algo_config=BNC_LOG_CPM_LINEAR_SVM_COND_ASSERT_AMBIG_NEG
#            )
#        ]
#        for c in commands:
#            shell(c)


# TODO This is the fall back HERE
#rule apply_trained_model_bnc_linear_svm_count_cond_prior_assert_ambig_neg:
#    input:
#        test_exp_list = '{env_dir}/data/experiment_lists/{exp_list}/experiment_list.json'.format(
#            env_dir=ENV_DIR,
#            exp_list='test_set.untampered_bulk_primary_cells_with_data'
#        ),
#        labelling_f = '{env_dir}/data/experiment_lists/{exp_list}/labelling.json'.format(
#            env_dir=ENV_DIR,
#            exp_list='test_set.untampered_bulk_primary_cells_with_data'
#        ),
#        model_f = '{env_dir}/data/experiment_lists/{train_exp_list}/trained_models/{algo_config}/model.pickle'.format(
#            env_dir=ENV_DIR,
#            train_exp_list='train_set.untampered_bulk_primary_cells_with_data',
#            algo_config=BNC_LOG_CPM_LINEAR_SVM_COND_ASSERT_AMBIG_NEG
#        )
#    output:
#        '{env_dir}/data/experiment_lists/{exp_list}/trained_on_another_experiment_list/{train_exp_list}/{algo_config}/prediction_results.json'.format(
#            env_dir=ENV_DIR,
#            exp_list='test_set.untampered_bulk_primary_cells_with_data',
#            train_exp_list='train_set.untampered_bulk_primary_cells_with_data',
#            algo_config=BNC_LOG_CPM_LINEAR_SVM_COND_ASSERT_AMBIG_NEG
#        )
#    run:
#        commands = [
#            'mkdir -p {env_dir}/data/experiment_lists/{exp_list}/trained_on_another_experiment_list/{train_exp_list}/{algo_config}'.format(
#                env_dir=ENV_DIR,
#                exp_list='test_set.untampered_bulk_primary_cells_with_data',
#                train_exp_list='train_set.untampered_bulk_primary_cells_with_data',
#                algo_config=BNC_LOG_CPM_LINEAR_SVM_COND_ASSERT_AMBIG_NEG
#            ),
#            'python2.7 {project_dir}/learning/apply_saved_model.py {{input.model_f}} {env_dir} {exp_list} -o {env_dir}/data/experiment_lists/{exp_list}/trained_on_another_experiment_list/{train_exp_list}/{algo_config}'.format(
#                env_dir=ENV_DIR,
#                project_dir=PROJECT_DIR,
#                exp_list='test_set.untampered_bulk_primary_cells_with_data',
#                train_exp_list='train_set.untampered_bulk_primary_cells_with_data',
#                algo_config=BNC_LOG_CPM_LINEAR_SVM_COND_ASSERT_AMBIG_NEG
#            )
#        ]
#        for c in commands:
#            shell(c)


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
            'python2.7 {project_dir}/learning/train_and_apply.py {project_dir}/learning/train_config/{env_name}/{algo_config}.json {env_dir} {train_exp_list} {exp_list} -r /scratch/mnbernstein/cell_type_phenotyping_tmp/train_and_save_artifacts -o {env_dir}/data/experiment_lists/{exp_list}/trained_on_another_experiment_list/{train_exp_list}/{algo_config}'.format(
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
            'python2.7 {project_dir}/learning/train_and_apply.py {project_dir}/learning/train_config/{env_name}/{algo_config}.json {env_dir} {train_exp_list} {exp_list} -r /scratch/mnbernstein/cell_type_phenotyping_tmp/train_and_save_artifacts -o {env_dir}/data/experiment_lists/{exp_list}/trained_on_another_experiment_list/{train_exp_list}/{algo_config}'.format(
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
#   Create the various experiment lists
#####################################################################

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


#####################################################################
#   Use the annotation to create the data set file
#####################################################################

rule apply_annotation:
    input:
        '{}/manage_data/annotations/from.cell_type_primary_cells_v1-4/cell_type_primary_cells_v1-4.annot_6.experiment_centric.json'.format(PROJECT_DIR)        
    output:
        '{}/data/data_set_metadata.json'.format(ENV_DIR) 
    shell:
        'python2.7 {}/manage_data/annotations/from.cell_type_primary_cells_v1-4/apply_annot.metasra_v1-4.annot_6.experiment_centric.py'.format(PROJECT_DIR) 
