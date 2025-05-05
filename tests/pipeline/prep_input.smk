### snakemake file for running setup


samples_tsv = pd.read_csv(config['samples_tsv'], sep='\t', index_col=0)

samples = list(samples_tsv.index) #list of samples



output_h5ad = expand(config['tidy_output_dir']+'/{sample}_vae_input.h5ad', sample=samples)



merged_h5ad = config['tidy_output_dir']+'/all_samples_merged_vae.h5ad'

rule tidy:
    params:
        cofactor = config['cofactor'],
        output_dir = config['tidy_output_dir']
    
    input:
        expression_tiff = lambda wildcards: samples_tsv.expression_tiff[wildcards.sample],
        mask_tiff = lambda wildcards: samples_tsv.mask_tiff[wildcards.sample],
        feature_data = config['feature_data'],
        #non_proteins = config['non_proteins'],
        
    
    output:
        config['tidy_output_dir']+'/{sample}_vae_input.h5ad'

    shell:
        "python vae_data_prep_sample.py --feature_data {input.feature_data} "
        #"--non_proteins {input.non_proteins} "
        "--cofactor {params.cofactor} "
        "--sample {wildcards.sample} --tiff {input.expression_tiff} --mask {input.mask_tiff} "
        "--output_dir {params.output_dir} "

rule merge:
    params:
        input_dir = config['tidy_output_dir'],

    input:
        output_h5ad,

    output:
        config['tidy_output_dir']+'/all_samples_merged_vae.h5ad',

    shell:
        "python vae_data_prep_concat.py --input_dir {params.input_dir} --output_h5ad {output} "