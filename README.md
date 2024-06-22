## ABGP 软件说明书目录

### 1. ABGP 简介

随着测序技术的发展和高通量基因型数据的产生，育种技术正由传统的“经验育种”逐步向“精准育种”转化。人工智能算法不需要预先的规则定义，通过数据和特征进行自主学习，擅长拟合数据中的非线性复杂关系，对于处理海量基因型数据更具有优势。将人工智能算法应用在基因组预测中，对于进一步提高重要表型育种值估计的准确性，加快遗传进展和育种进程具有重要意义。  
Artificial Intelligence Breeding Pipeline(ABGP)是一款集特征选择、模型构建、速度优化、参数调节、位点可解释性于一体的AI育种软件，可以融合先验信息、场季效应、性别、批次等多种协变量信息，针对质量性状和数量性状，包含多种不同的AI算法模型，支持cpu、gpu计算的同时加入多进程、多线程、最优化迭代、网格搜索、麻雀搜索、SNP位点可解释性分析等模块。

整体具有以下特征：
1. 无需重算亲缘关系矩阵：ABGP 通过直接使用特征数据进行训练和预测，避免了亲缘关系矩阵的计算和扩展问题。
2. 大幅降低内存消耗：通过优化数据处理和模型训练流程，ABGP 显著降低了内存消耗。
3. 大幅提升预测速度：ABGP 支持数据读取加速和 GPU 模型训练加速，在处理大规模数据时表现尤为出色。
4. 灵活性和适应性：机器学习方法可以处理不同类型的数据和任务，包括回归、分类以及具有复杂非线性关系的数据。这使得它们在处理复杂的基因组数据时更加灵活和高效。
5. 特征选择与重要性分析：机器学习模型可以自动进行特征选择，并通过如SHAP（SHapley Additive exPlanations）等方法评估特征重要性，提供对预测结果的深入理解和解释。而GBLUP方法则主要依赖于线性假设，特征选择和重要性分析相对较弱。
6. 高性能和精度：通过使用先进的模型，可以显著提高预测的精度和性能。这些模型通过高效的梯度提升技术，能够在处理大规模数据和复杂模型时表现出色。
7. 自动化调参：机器学习软件集成了网格搜索和智能算法（如SSA）进行自动化调参，确保模型在不同数据集上的最佳表现。而GBLUP方法通常需要手动调参，效率较低。
8. 处理协变量：ABGP可以直接处理数据中的分类变量，自动进行编码和处理，而传统GBLUP方法在处理分类变量时较为复杂，需要额外的预处理步骤。
此外，ABGP还通过麻雀搜索等多种参数优化方法进一步提升预测准确性，可以根据 snp 在模型中的权重挖掘与表型相关的 snp 位点。通过多种方法评估SNP位点重要性，一般 snp位点的权重越大，与表型的相关性越强。此外，通过shap理论进行模型可解释性的输出，包括自定义选择特征位点进行重要性排序、基于单样本、多样本与特征结合进行局部和全局性交互解释，并进行可视化展示等，用户可自定义不同特征及样本之间的交互关系进行进一步研究。建议用户将 ABGP不同方法提供的各 snp 的权重与 GWAS 结果对比参考，排除各自方法中的假阳性结果，提高可信度。
ABGP 在大数据时代，应对基因组数据分析需求方面展现了强大的优势，能够高效、准确地进行大规模数据的预测和分析。为AI育种提供了新的工具和软件平台。

### 2. 安装指南

#### 1. 克隆项目仓库

克隆项目的 GitHub 仓库到您的本地机器上。打开终端并执行以下命令：

```sh
git clone https://github.com/leiweiucas/ABGP
cd ABGP

