from numpy.core import machar
import pandas as pd
import numpy as np
import re
import sys

from pandas.core.algorithms import isin, value_counts
import pypeta
from pypeta import Peta
from collections import defaultdict


class SampleIdError(RuntimeError):
    def __init__(self, sample_id: str, message: str):
        self.sample_id = sample_id
        self.message = message


class NotNumericSeriesError(RuntimeError):
    def __init__(self, message: str):
        self.message = message


class UnknowSelectionTypeError(RuntimeError):
    def __init__(self, message: str):
        self.message = message


class NotInColumnError(RuntimeError):
    def __init__(self, message: str):
        self.message = message


class GenesRelationError(RuntimeError):
    def __init__(self, message: str):
        self.message = message


class VariantUndefinedError(RuntimeError):
    def __init__(self, message: str):
        self.message = message


class ListsUnEqualLengthError(RuntimeError):
    def __init__(self, message: str):
        self.message = message


class CDx_Data():
    """[summary]
    """
    def __init__(self,
                 mut_df: pd.DataFrame = None,
                 cli_df: pd.DataFrame = None,
                 cnv_df: pd.DataFrame = None,
                 sv_df: pd.DataFrame = None):
        """Constructor method with DataFrames

        Args:
            mut_df (pd.DataFrame, optional): SNV and InDel info. Defaults to None.
            cli_df (pd.DataFrame, optional): Clinical info. Defaults to None.
            cnv_df (pd.DataFrame, optional): CNV info. Defaults to None.
            sv_df (pd.DataFrame, optional): SV info. Defaults to None.
        """

        self.mut = mut_df
        self.cnv = cnv_df
        self.sv = sv_df

        if not cli_df is None:
            self.cli = cli_df
        else:
            self._set_cli()
        self.crosstab = self.get_crosstab()

    def from_PETA(self, token: str, json_str: str):
        """Retrieve CDx data from BGI-PETA database. 

        Args:
            token (str): Effective token for BGI-PETA database
            json_str (str): The json format restrictions communicating to the database
        """
        peta = Peta(token=token, host='https://peta.bgi.com/api')
        peta.set_data_restriction_from_json_string(json_str)

        self.cli = peta.fetch_clinical_data()
        self.mut = peta.fetch_mutation_data()
        self.cnv = peta.fetch_cnv_data()
        self.sv = peta.fetch_sv_data()

        self.crosstab = self.get_crosstab()

    def from_file(self,
                  mut_f: str = None,
                  cli_f: str = None,
                  cnv_f: str = None,
                  sv_f: str = None):
        """Get CDx data from files.

        Args:
            mut_f (str, optional): File as NCBI MAF format contains SNV and InDel. Defaults to None.
            cli_f (str, optional): File name contains clinical info. Defaults to None.
            cnv_f (str, optional): File name contains CNV info. Defaults to None.
            sv_f (str, optional): File name contains SV info. Defaults to None.
        """
        if not mut_f is None:
            self.mut = pd.read_csv(mut_f, sep='\t')

        if not cnv_f is None:
            self.cnv = pd.read_csv(cnv_f, sep='\t')

        if not sv_f is None:
            self.sv = pd.read_csv(sv_f, sep='\t')

        if not cli_f is None:
            self.cli = pd.read_csv(cli_f, sep='\t')
        else:
            self._set_cli()

        self.crosstab = self.get_crosstab()

    def _set_cli(self):
        """Set the cli attribute, generate a void DataFrame when it is not specified. 
        """
        sample_id_series = []
        if not self.mut is None:
            sample_id_series.append(
                self.mut['Tumor_Sample_Barcode'].drop_duplicates())

        if not self.cnv is None:
            sample_id_series.append(
                self.cnv['Tumor_Sample_Barcode'].drop_duplicates())

        if not self.sv is None:
            sample_id_series.append(
                self.sv['Tumor_Sample_Barcode'].drop_duplicates())

        if len(sample_id_series) > 0:
            self.cli = pd.DataFrame({
                'sampleId': pd.concat(sample_id_series)
            }).drop_duplicates()
        else:
            self.cli = None

    def get_crosstab(self) -> pd.DataFrame:
        """Generate a Gene vs. Sample_id cross table.

        Raises:
            SampleIdError: Sample id from the mut, cnv or sv which not exsits in the cli table.

        Returns:
            pd.DataFrame: CDx_Data.
        """
        if self.cli is None:
            return pd.DataFrame([])

        sub_dfs = []
        # cli
        cli = self.cli.copy().set_index('sampleId').T
        cli['track_type'] = 'CLINICAL'
        sub_dfs.append(cli)

        # mut. represent by cHgvs, joined by '|' for mulitple hit
        if not self.mut is None:
            genes = self.mut['Hugo_Symbol'].drop_duplicates()

            mut = pd.DataFrame(np.zeros((len(genes), len(cli.columns))),
                               index=genes,
                               columns=cli.columns)
            mut = mut.replace({0: np.nan})
            mut['track_type'] = 'MUTATIONS'

            # 3 columns, Hugo_symbol,Tumor_Sample_Barcode,cHgvs.
            for _, row in self.mut.iterrows():
                try:
                    if pd.isnull(mut.loc[row['Hugo_Symbol'],
                                         row['Tumor_Sample_Barcode']]):
                        mut.loc[row['Hugo_Symbol'], row[
                            'Tumor_Sample_Barcode']] = row['HGVSp_Short']
                    else:
                        mut.loc[row['Hugo_Symbol'], row[
                            'Tumor_Sample_Barcode']] += f'|{row["HGVSp_Short"]}'
                except:
                    raise SampleIdError(row['Tumor_Sample_Barcode'],
                                        'not exists in clinical table')

            sub_dfs.append(mut)

        # cnv. represent by gain or loss. at first use the virtual column "status"
        if not self.cnv is None:
            genes = self.cnv['Hugo_Symbol'].drop_duplicates()
            cnv = pd.DataFrame(np.zeros((len(genes), len(cli.columns))),
                               index=genes,
                               columns=cli.columns)
            cnv = cnv.replace({0: np.nan})
            cnv['track_type'] = 'CNV'

            for _, row in self.cnv.iterrows():
                try:
                    cnv.loc[row['Hugo_Symbol'],
                            row['Tumor_Sample_Barcode']] = row['status']
                except:
                    raise SampleIdError(row['Tumor_Sample_Barcode'],
                                        'not exists in clinical table')

            sub_dfs.append(cnv)

        # sv. represent by gene1 and gene2 combination.
        if not self.sv is None:
            genes = self.sv['gene1'].drop_duplicates()
            sv = pd.DataFrame(np.zeros((len(genes), len(cli.columns))),
                              index=genes,
                              columns=cli.columns)
            sv = sv.replace({0: np.nan})
            sv['track_type'] = 'FUSION'

            for _, row in self.sv.iterrows():
                fusion_symbol = f'{row["gene1"]}-{row["gene2"]}'
                try:
                    if pd.isnull(
                            sv.loc[row['gene1'], row['Tumor_Sample_Barcode']]):
                        sv.loc[row['gene1'],
                               row['Tumor_Sample_Barcode']] = fusion_symbol
                    else:
                        sv.loc[row['gene1'], row[
                            'Tumor_Sample_Barcode']] = f'|{fusion_symbol}'
                except:
                    raise SampleIdError(row['Tumor_Sample_Barcode'],
                                        'not exists in clinical table')

            sub_dfs.append(sv)

        return pd.concat(sub_dfs)

    #如何构建通用的选择接口，通过变异、基因、癌种等进行选择，并支持“或”和“且”的逻辑运算
    #该接口至关重要，对变异入选条件的选择会影响到crosstab，
    #选择后返回一个新的CDX_Data对象
    def select(self, conditions: dict = {}, update=True):
        """A universe interface to select data via different conditions.

        Args:
            conditions (dict, optional): Each key represent one column`s name of the CDx_Data attributes. Defaults to {}.
            update (bool, optional): [description]. Defaults to True.
        """
        return self

    # 数据选择的辅助函数
    def _numeric_selector(self, ser: pd.Series, range: str) -> pd.Series:
        """Compute a comparition expression on a numeric Series

        Args:
            ser (pd.Series): Numeric Series.
            range (str): comparition expression like 'x>5'. 'x' is mandatory and represent the input. 

        Raises:
            NotNumericSeriesError: Input Series`s dtype is not a numeric type.

        Returns:
            pd.Series: Series with boolean values.
        """
        if ser.dtype == 'object':
            raise NotNumericSeriesError(f'{ser.name} is not numeric')

        #return ser.map(lambda x: eval(re.sub(r'x', str(x), range)))
        return eval(re.sub(r'x', 'ser', range))

    def _catagory_selector(self, ser: pd.Series, range: list) -> pd.Series:
        """Return True if the Series` value in the input range list.

        Args:
            ser (pd.Series): Catagory Series.
            range (list): List of target options.

        Returns:
            pd.Series: Series with boolean values
        """
        return ser.isin(range)

    def _selector(self, df: pd.DataFrame, selections: dict) -> pd.DataFrame:
        """Filter the input DataFrame via the dict of conditions.

        Args:
            df (pd.DataFrame): Input.
            selections (dict): Dict format of conditions like "{'Cancer_type':['lung','CRC'],'Age':'x>5'}".
                The keys represent a column in the input DataFrame.
                The list values represent a catagory target and str values represent a numeric target.

        Raises:
            NotInColumnError: Key in the dict is not in the df`s columns.
            UnknowSelectionTypeError: The type of value in the dict is not str nor list.

        Returns:
            pd.DataFrame: Filterd DataFrame
        """
        columns = df.columns
        for key, value in selections.items():
            if key not in columns:
                raise NotInColumnError(f'{key} is not in the columns')

            if isinstance(value, str):
                df = df[self._numeric_selector(df[key], value)]
            elif isinstance(value, list):
                df = df[self._catagory_selector(df[key], value)]
            else:
                raise UnknowSelectionTypeError(
                    f'{selections} have values not str nor list')

        return df

    def _fuzzy_id(self, regex: re.Pattern, text: str) -> str:
        """transform a sample id into fuzzy mode according the regex pattern

        Args:
            regex (re.Pattern): The info retains are in the capture patterns
            text (str): input sample id

        Returns:
            str: fuzzy mode sample id
        """
        matches = regex.findall(text)
        if matches:
            text = '_'.join(matches[0])

        return text

    def select_by_sample_ids(self,
                             sample_ids: list,
                             fuzzy: bool = False,
                             regex_str: str = r'(\d+)[A-Z](\d+)',
                             study_ids: list = []):
        """Select samples via a list of sample IDs.

        Args:
            sample_ids (list): sample ids list.
            fuzzy (bool): fuzzy mode.
            regex_str (str): The match principle for fuzzy match. The info in the regex capture patterns must be matched for a certifired record. Default for r'(\d+)[A-Z](\d+)'.  
            study_ids: (list): The corresponding study id of each sample ids. Length of sample_ids and study_ids must be the same.

        Raises:
            ListsUnEqualLengthError: Length of sample_ids and study_ids are not equal.

        Returns:
            CDx: CDx object of selected samples.
        """
        if fuzzy:
            regex = re.compile(regex_str)

            # fuzzy the input ids
            target_ids = []
            fuzzy_to_origin = {}
            transform = lambda x: self._fuzzy_id(regex, x)
            for sample_id in sample_ids:
                fuzzy_sample_id = self._fuzzy_id(regex, sample_id)
                fuzzy_to_origin[fuzzy_sample_id] = sample_id
                target_ids.append(fuzzy_sample_id)
        else:
            target_ids = sample_ids
            transform = lambda x: x

        # match
        sample_id_bool = self.cli['sampleId'].map(transform).isin(target_ids)
        if study_ids:
            if len(study_ids) != len(sample_ids):
                raise ListsUnEqualLengthError('Error')

            sub_cli_df = self.cli[sample_id_bool]
            study_id_bool = sub_cli_df.apply(
                lambda x: x['studyId'] == study_ids[target_ids.index(
                    transform(x['sampleId']))],
                axis=1)
            sample_id_bool = sample_id_bool & study_id_bool

        # construct new CDx_Data object
        # CDx_Data always have a cli
        cli_df = self.cli[sample_id_bool].copy()

        # add a column of query ids for fuzzy match
        if fuzzy:
            cli_df['queryId'] = cli_df['sampleId'].map(
                lambda x: fuzzy_to_origin[transform(x)])

        if not self.mut is None:
            mut_df = self.mut[self.mut['Tumor_Sample_Barcode'].isin(
                cli_df['sampleId'])]
        else:
            mut_df = None

        if not self.cnv is None:
            cnv_df = self.cnv[self.cnv['Tumor_Sample_Barcode'].isin(
                cli_df['sampleId'])]
        else:
            cnv_df = None

        if not self.sv is None:
            sv_df = self.sv[self.sv['Tumor_Sample_Barcode'].isin(
                cli_df['sampleId'])]
        else:
            sv_df = None

        return CDx_Data(cli_df=cli_df,
                        mut_df=mut_df,
                        cnv_df=cnv_df,
                        sv_df=sv_df)

    #
    def set_mut_eligibility(self, **kwargs):
        """Set threshold for SNV/InDels to regrard as a positive sample

        Raises:
            VariantUndefinedError: mut info not provided by user.

        Returns:
            CDx_Data: CDx_Data object
        """
        if self.mut is None:
            raise VariantUndefinedError(f'mut variant undefied')

        self.mut = self._selector(self.mut, kwargs)
        return CDx_Data(cli_df=self.cli,
                        mut_df=self.mut,
                        cnv_df=self.cnv,
                        sv_df=self.sv)

    def set_cnv_eligibility(self, **kwargs):
        """Set threshold for CNV to regrard as a positive sample.

        Raises:
            VariantUndefinedError: cnv info not provided by user.

        Returns:
            CDx_Data: CDx_Data object.
        """
        if self.cnv is None:
            raise VariantUndefinedError(f'cnv variant undefied')

        self.cnv = self._selector(self.cnv, kwargs)
        return CDx_Data(cli_df=self.cli,
                        mut_df=self.mut,
                        cnv_df=self.cnv,
                        sv_df=self.sv)

    def set_sv_eligibility(self, **kwargs):
        """Set threshold for SV to regrard as a positive sample.

        Raises:
            VariantUndefinedError: SV info not provided by user.

        Returns:
            CDx_Data: CDx_Data object.
        """
        if self.sv is None:
            raise VariantUndefinedError(f'sv variant undefied')

        self.sv = self._selector(self.sv, kwargs)
        return CDx_Data(cli_df=self.cli,
                        mut_df=self.mut,
                        cnv_df=self.cnv,
                        sv_df=self.sv)

    # 指定一个列名，再指定范围。离散型用数组，数值型
    # attrdict={'Cancer_type':['lung','CRC'],'Age':'x>5'}
    def select_samples_by_clinical_attributes2(self, attr_dict: dict):
        """Select samples via a set of conditions corresponding to the columns in the cli DataFrame.

        Args:
            attr_dict (dict): Dict format of conditions like "{'Cancer_type':['lung','CRC'],'Age':'x>5'}".
                The keys represent a column in the input DataFrame.
                The list values represent a catagory target and str values represent a numeric target.

        Returns:
            CDx: CDx object of selected samples.
        """
        cli_df = self._selector(self.cli, attr_dict)
        return self.select_by_sample_ids(cli_df['sampleId'])

    def select_samples_by_clinical_attributes(self, **kwargs):
        """Select samples via a set of conditions corresponding to the columns in the cli DataFrame.

        Args:
            Keywords arguments with each key represent a column in the input DataFrame.
                like "Cancer_type=['lung','CRC'], Age='x>5'"
                The list values represent a catagory target and str values represent a numeric target.

        Returns:
            CDx: CDx object of selected samples.
        """
        cli_df = self._selector(self.cli, kwargs)
        return self.select_by_sample_ids(cli_df['sampleId'])

    # 对阳性样本进行选取。基因组合，且或关系，chgvs和ghgvs，基因系列如MMR、HR等
    # 基因组合可以做为入参数组来传入
    def select_samples_by_mutate_genes(
            self,
            genes: list,
            variant_type: list = ['MUTATIONS', 'CNV', 'SV'],
            how='or'):
        """Select sample via positve variant genes.

        Args:
            genes (list): Gene Hugo names.
            variant_type (list, optional): Combination of MUTATIONS, CNV and SV. Defaults to ['MUTATIONS', 'CNV', 'SV'].
            how (str, optional): 'and' for variant in all genes, 'or' for variant in either genes. Defaults to 'or'.

        Raises:
            GenesRelationError: Value of how is not 'and' nor 'or'.

        Returns:
            CDx: CDx object of selected samples.
        """
        variant_crosstab = self.crosstab[
            self.crosstab['track_type'] != 'CLINICAL']

        if variant_type:
            variant_crosstab = variant_crosstab[
                variant_crosstab['track_type'].isin(variant_type)]

        if how == 'or':
            is_posi_sample = variant_crosstab.reindex(
                index=genes).apply(lambda x: any(pd.notnull(x)))
        elif how == 'and':
            is_posi_sample = variant_crosstab.reindex(
                index=genes).apply(lambda x: all(pd.notnull(x)))
        else:
            raise GenesRelationError(
                f'value of "how" must be "or" or "and", here comes "{how}"')

        # the last column is "track_type"
        sample_ids = is_posi_sample[is_posi_sample][:-1].index

        return self.select_by_sample_ids(sample_ids)

    # Analysis
    def tableone(self):
        pass

    def pathway(self):
        pass

    def pinpoint(self):
        pass

    def oncoprint(self):
        pass

    def survival(self):
        pass

    def positive_rate(self):
        pass

    def sample_size_by_time(self):
        pass
