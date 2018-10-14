# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/)
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

<!--

============== Guiding Principles ==============

* Changelogs are for humans, not machines.
* There should be an entry for every single version.
* The same types of changes should be grouped.
* Versions and sections should be linkable.
* The latest version comes first.
* The release date of each version is displayed.
* Mention whether you follow Semantic Versioning.

============== Types of changes (keep the order) ==============

* `Added` for new features.
* `Changed` for changes in existing functionality.
* `Deprecated` for soon-to-be removed features.
* `Removed` for now removed features.
* `Fixed` for any bug fixes.
* `Security` in case of vulnerabilities.
* `Dependencies Update` in case of vulnerabilities.
* `Contributors` to thank the contributors that worked on this PR.

============== How To Update The Changelog for a New Release ==============

** Always Keep The Unreleased On Top **

To release a new version, please update the changelog as followed:
1. Rename the `Unreleased` Section to the Section Number
2. Recreate an `Unreleased` Section on top
3. Update the links at the very bottom

======================= START: TEMPLATE TO KEEP IN CASE OF NEED ===================

** DO NOT MODIFY THIS SECTION ! **

## [Unreleased]

### Added

### Changed

### Dependencies Update

### Deprecated

### Fixed

### Removed

### Security

### Contributors

** DO NOT MODIFY THIS SECTION ! **

======================= END: TEMPLATE TO KEEP IN CASE OF NEED ===================

-->

<!-- YOU CAN EDIT FROM HERE -->

## [Unreleased]

### Added
- Layer:
  - Release `GroupNormLayer` (PR #850)
- Tests:
  - `test_model_compilednetwork.py` has been introduced to test the class `tl.models.CompiledNetwork` (PR #755).
  - `test_network_custom_input_layers.py` has been introduced to test the class `tl.networks.CustomModel` (PR #755).
  - `test_network_custom_multiple_inputs.py` has been introduced to test the class `tl.networks.CustomModel` (PR #755).
  - `test_network_custom_multiple_outputs.py` has been introduced to test the class `tl.networks.CustomModel` (PR #755).
  - `test_network_custom_1d.py` has been introduced to test the class `tl.networks.CustomModel` with 1D data inputs (PR #755).
  - `test_network_custom_2d.py` has been introduced to test the class `tl.networks.CustomModel` with 2D data inputs (PR #755).
  - `test_network_custom_rnn.py` has been introduced to test the class `tl.networks.CustomModel` with sequential data inputs (PR #755).
  - `test_network_sequential_1d.py` has been introduced to test the class `tl.networks.Sequential` with 1D data Layers (PR #755).
  - `test_network_sequential_2d.py` has been introduced to test the class `tl.networks.Sequential` with 2D data Layers (PR #755).
  - `test_network_sequential_3d.py` has been introduced to test the class `tl.networks.Sequential` with 3D data Layers (PR #755).
  - `test_network_sequential_rnn.py` has been introduced to test the class `tl.networks.Sequential` with 3D data Layers (PR #755).

### Changed
- API:
  - `tl.models` has been renamed to `tl.nets` to comply with TF Slim naming convention: https://github.com/tensorflow/models/tree/master/research/slim (PR #755)
  - `layer.all_params` renamed `layer.all_weights` to reduce confusion (params != hyperparameters, params = weights) (PR #755)
  - `layer.count_params()` renamed `layer.count_weights()` to reduce confusion (params != hyperparameters, params = weights) (PR #755)
  - `layer.get_all_params()` renamed `layer.get_all_weights()` to reduce confusion (params != hyperparameters, params = weights) (PR #755)
  - `layer.print_params()` renamed `layer.print_weights()` to reduce confusion (params != hyperparameters, params = weights) (PR #755)
  - `tl.layers.ROIPoolingLayer` has been moved to `tl.layers.contrib.ROIPoolingLayer` (PR #755)
  - `tl.layers.QuanDenseLayer` renamed to `tl.layers.QuantizedDense` (PR #755)
  - `tl.layers.QuanDenseLayerWithBN` renamed to `tl.layers.QuantizedDenseWithBN` (PR #755)
  - `tl.layers.QuanConv2d` renamed to `tl.layers.QuantizedConv2d` (PR #755)
  - `tl.layers.QuanConv2dWithBN` renamed to `tl.layers.QuantizedConv2dWithBN` (PR #755)

### Dependencies Update
- yapf>=0.22,<0.24 => yapf>=0.22,<0.25 (PR #829)
- sphinx>=1.7,<1.8 => sphinx>=1.7,<1.9 (PR #842)
- matplotlib>=2.2,<2.3 => matplotlib>=2.2,<3.1 (PR #845)
- scikit-learn>=0.19,<0.20 => scikit-learn>=0.19,<0.21 (PR #851)
- tensorflow>=1.6,<1.11 => tensorflow>=1.6,<1.12 (PR #853)
- tqdm>=4.23,<4.26 => tqdm>=4.23,<4.27 (PR #862)

### Deprecated

### Fixed
- Correct offset calculation in `tl.prepro.transform_matrix_offset_center` (PR #855)

### Removed
- `tl.layers.EstimatorLayer` has been removed in favor of `tl.layers.LambdaLayer` (PR #755)
- `tl.layers.ReconLayer` has been removed in favor of `tl.layers.DenseLayer` (#755)
- `tl.layers.MultiplexerLayer` has been removed in favor of eager execution mode (#755)
- `tl.layers.AtrousConv1dLayer` has been removed in favor of `Conv1d` with dilation.
- `tl.layers.AtrousConv2dLayer` has been removed in favor of `Conv2d` with dilation.

### Security

### Contributors
- @2wins: #850 #855
- @DEKHTIARJonathan: #755 #853
- @zsdonghao: #755

## [1.10.1] - 2018-09-07

### Added
- unittest `tests\test_timeout.py` has been added to ensure the network creation process does not freeze.

### Changed
 - remove 'tensorboard' param, replaced by 'tensorboard_dir' in `tensorlayer/utils.py` with customizable tensorboard directory (PR #819)

### Removed
- TL Graph API removed. Memory Leaks Issues with Graph API, will be fixed and integrated in TL 2.0 (PR #818)

### Fixed
- Issue #817 fixed: TL 1.10.0 - Memory Leaks and very slow network creation.

### Dependencies Update
- autopep8>=1.3,<1.4 => autopep8>=1.3,<1.5 (PR #815)
- pytest-cov>=2.5,<2.6 => pytest-cov>=2.5,<2.7 (PR #820)
- pytest>=3.6,<3.8 => pytest>=3.6,<3.9 (PR #823)
- imageio>=2.3,<2.4 => imageio>=2.3,<2.5 (PR #823)

### Contributors
- @DEKHTIARJonathan: #815 #818 #820 #823
- @ndiy: #819
- @zsdonghao: #818

## [1.10.0] - 2018-09-02

### Added

- API:
  - Add `tl.model.vgg19` (PR #698)
  - Add `tl.logging.contrib.hyperdash` (PR #739)
  - Add `tl.distributed.trainer` (PR #700)
  - Add `prefetch_buffer_size` to the `tl.distributed.trainer` (PR #766)
  - Add `tl.db.TensorHub` (PR ＃751)
  - Add `tl.files.save_graph` (PR ＃751)
  - Add `tl.files.load_graph_` (PR ＃751)
  - Add `tl.files.save_graph_and_params` (PR ＃751)
  - Add `tl.files.load_graph_and_params` (PR ＃751)
  - Add `tl.prepro.keypoint_random_xxx` (PR #787)
- Documentation:
  - Add binary, ternary and dorefa links (PR #711)
  - Update input scale of VGG16 and VGG19 to 0~1 (PR #736)
  - Update database (PR ＃751)
- Layer:
  - Release SwitchNormLayer (PR #737)
  - Release QuanConv2d, QuanConv2dWithBN, QuanDenseLayer, QuanDenseLayerWithBN (PR#735)
  - Update Core Layer to support graph (PR ＃751)
  - All Pooling layers support `data_format` (PR #809)
- Setup:
  - Creation of installation flaggs `all_dev`, `all_cpu_dev`, and `all_gpu_dev` (PR #739)
- Examples:
  - change folder struction (PR #802)
  - `tutorial_models_vgg19` has been introduced to show how to use `tl.model.vgg19` (PR #698).
  - fix bug of `tutorial_bipedalwalker_a3c_continuous_action.py` (PR #734, Issue #732)
  - `tutorial_models_vgg16` and `tutorial_models_vgg19` has been changed the input scale from [0,255] to [0,1](PR #710)
  - `tutorial_mnist_distributed_trainer.py` and `tutorial_cifar10_distributed_trainer.py` are added to explain the uses of Distributed Trainer (PR #700)
  - add `tutorial_quanconv_cifar10.py` and `tutorial_quanconv_mnist.py` (PR #735)
  - add `tutorial_work_with_onnx.py`(PR #775)
- Applications:
  - [Arbitrary Style Transfer in Real-time with Adaptive Instance Normalization](https://arxiv.org/abs/1703.06868) (PR #799)

### Changed

  - function minibatches changed to avoid wasting samples.(PR #762)
  - all the input scale in both vgg16 and vgg19 has been changed the input scale from [0,255] to [0,1](PR #710)
  - Dockerfiles merged and refactored into one file (PR #747)
  - LazyImports move to the most **top level** imports as possible (PR #739)
  - some new test functions have been added in `test_layers_convolution.py`, `test_layers_normalization.py`, `test_layers_core.py` (PR #735)
  - documentation now uses mock imports reducing the number of dependencies to compile the documentation (PR #785)
  - fixed and enforced pydocstyle D210, D200, D301, D207, D403, D204, D412, D402, D300, D208 (PR #784)

### Deprecated

  - `tl.logging.warn` has been deprecated in favor of `tl.logging.warning` (PR #739)

### Removed

  - `conv_layers()`  has been removed in both vgg16 and vgg19(PR #710)
  - graph API (PR #818)

### Fixed

- import error caused by matplotlib on OSX (PR #705)
- missing import in tl.prepro (PR #712)
- Dockerfiles import error fixed - issue #733 (PR #747)
- Fix a typo in `absolute_difference_error` in file: `tensorlayer/cost.py` - Issue #753 (PR #759)
- Fix the bug of scaling the learning rate of trainer (PR #776)
- log error instead of info when npz file not found. (PR #812)

### Dependencies Update

- tensorflow>=1.8,<1.9 => tensorflow>=1.6,<1.11 (PR #739 and PR #798)
- tensorflow-gpu>=1.8,<1.9 => tensorflow-gpu>=1.6,<1.11 (PR #739 and PR #798)
- numpy>=1.14,<1.15 => numpy>=1.14,<1.16 (PR #754)
- pymongo>=3.6,<3.7 => pymongo>=3.6,<3.8 (PR #750)
- pytest>=3.6,<3.7 => tqdm>=3.6,<3.8 (PR #798)
- pytest-xdist>=1.22,<1.23 => pytest-xdist>=1.22,<1.24 (PR #805 and #806)
- tqdm>=4.23,<4.25 => tqdm>=4.23,<4.26 (PR #798)
- yapf>=0.21,<0.22 => yapf>=0.22,<0.24 (PR #798 #808)

### Contributors

- @DEKHTIARJonathan: #739 #747 #750 #754
- @lgarithm: #705 #700
- @OwenLiuzZ: #698 #710 #775 #776
- @zsdonghao: #711 #712 #734 #736 #737 #700 #751 #809 #818
- @luomai: #700 #751 #766 #802
- @XJTUWYD: #735
- @mutewall: #735
- @thangvubk: #759
- @JunbinWang: #796
- @boldjoel: #787

## [1.9.1] - 2018-07-30

### Fixed
- Issue with tensorflow 1.10.0 fixed

## [1.9.0] - 2018-06-16

### Added

- API:
  - `tl.alphas` and `tl.alphas_like` added following the tf.ones/zeros and tf.zeros_like/ones_like (PR #580)
  - `tl.lazy_imports.LazyImport` to import heavy libraries only when necessary (PR #667)
  - `tl.act.leaky_relu6` and `tl.layers.PRelu6Layer` have been deprecated (PR #686)
  - `tl.act.leaky_twice_relu6` and `tl.layers.PTRelu6Layer` have been deprecated (PR #686)
- CI Tool:
  - [Stale Probot](https://github.com/probot/stale) added to clean stale issues (PR #573)
  - [Changelog Probot](https://github.com/mikz/probot-changelog) Configuration added (PR #637)
  - Travis Builds now handling a matrix of TF Version from TF==1.6.0 to TF==1.8.0 (PR #644)
  - CircleCI added to build and upload Docker Containers for each PR merged and tag release (PR #648)
- Decorator:
  - `tl.decorators` API created including `deprecated_alias` and `private_method` (PR #660)
  - `tl.decorators` API enriched with `protected_method` (PR #675)
  - `tl.decorators` API enriched with `deprecated` directly raising warning and modifying documentation (PR #691)
- Docker:
  - Containers for each release and for each PR merged on master built (PR #648)
  - Containers built in the following configurations (PR #648):
    - py2 + cpu
    - py2 + gpu
    - py3 + cpu
    - py3 + gpu
- Documentation:
  - Clean README.md (PR #677)
  - Release semantic version added on index page (PR #633)
  - Optimizers page added (PR #636)
  - `AMSGrad` added on Optimizers page added (PR #636)
- Layer:
  - ElementwiseLambda added to use custom function to connect multiple layer inputs (PR #579)
  - AtrousDeConv2dLayer added (PR #662)
  - Fix bugs of using `tf.layers` in CNN (PR #686)
- Optimizer:
  - AMSGrad Optimizer added based on `On the Convergence of Adam and Beyond (ICLR 2018)` (PR #636)
- Setup:
  - Creation of installation flaggs `all`, `all_cpu`, and `all_gpu` (PR #660)
- Test:
  - `test_utils_predict.py` added to reproduce and fix issue #288 (PR #566)
  - `Layer_DeformableConvolution_Test` added to reproduce issue #572 with deformable convolution (PR #573)
  - `Array_Op_Alphas_Test` and `Array_Op_Alphas_Like_Test` added to test `tensorlayer/array_ops.py` file (PR #580)
  - `test_optimizer_amsgrad.py` added to test `AMSGrad` optimizer (PR #636)
  - `test_logging.py` added to insure robustness of the logging API (PR #645)
  - `test_decorators.py` added (PR #660)
  - `test_activations.py` added (PR #686)
- Tutorials:
  - `tutorial_tfslim` has been introduced to show how to use `SlimNetsLayer` (PR #560).
  - add the following to all tutorials (PR #697):
    ```python
    tf.logging.set_verbosity(tf.logging.DEBUG)
    tl.logging.set_verbosity(tl.logging.DEBUG)
    ```

### Changed

- Tensorflow CPU & GPU dependencies moved to separated requirement files in order to allow PyUP.io to parse them (PR #573)
- The document of LambdaLayer for linking it with ElementwiseLambda (PR #587)
- RTD links point to stable documentation instead of latest used for development (PR #633)
- TF Version older than 1.6.0 are officially unsupported and raises an exception (PR #644)
- README.md Badges Updated with Support Python and Tensorflow Versions (PR #644)
- TL logging API has been consistent with TF logging API and thread-safe (PR #645)
- Relative Imports changed for absolute imports (PR #657)
- `tl.files` refactored into a directory with numerous files (PR #657)
- `tl.files.voc_dataset` fixed because of original Pascal VOC website was down (PR #657)
- extra requirements hidden inside the library added in the project requirements (PR #657)
- requirements files refactored in `requirements/` directory (PR #657)
- README.md and other markdown files have been refactored and cleaned. (PR #639)
- Ternary Convolution Layer added in unittest (PR #658)
- Convolution Layers unittests have been cleaned & refactored (PR #658)
- All the tests are now using a DEBUG level verbosity when run individualy (PR #660)
- `tf.identity` as activation is **ignored**, thus reducing the size of the graph by removing useless operation (PR #667)
- argument dictionaries are now checked and saved within the `Layer` Base Class (PR #667)
- `Layer` Base Class now presenting methods to update faultlessly `all_layers`, `all_params`, and `all_drop` (PR #675)
- Input Layers have been removed from `tl.layers.core` and added to `tl.layers.inputs` (PR #675)
- Input Layers are now considered as true layers in the graph (they represent a placeholder), unittests have been updated (PR #675)
- Layer API is simplified, with automatic feeding `prev_layer` into `self.inputs` (PR #675)
- Complete Documentation Refactoring and Reorganization (namely Layer APIs) (PR #691)

### Deprecated

- `tl.layers.TimeDistributedLayer` argurment `args` is deprecated in favor of `layer_args` (PR #667)
- `tl.act.leaky_relu` have been deprecated in favor of `tf.nn.leaky_relu` (PR #686)

### Removed

- `assert()` calls remove and replaced by `raise AssertionError()` (PR #667)
- `tl.identity` is removed, not used anymore and deprecated for a long time (PR #667)
- All Code specific to `TF.__version__ < "1.6"` have been removed (PR #675)

### Fixed

- Issue #498 - Deprecation Warning Fix in `tl.layers.RNNLayer` with `inspect` (PR #574)
- Issue #498 - Deprecation Warning Fix in `tl.files` with truth value of an empty array is ambiguous (PR #575)
- Issue #565 related to `tl.utils.predict` fixed - `np.hstack` problem in which the results for multiple batches are stacked along `axis=1` (PR #566)
- Issue #572 with `tl.layers.DeformableConv2d` fixed (PR #573)
- Issue #664 with `tl.layers.ConvLSTM` fixed (PR #676)
- Typo of the document of ElementwiseLambda (PR #588)
- Error in `tl.layers.TernaryConv2d` fixed - self.inputs not defined (PR #658)
- Deprecation warning fixed in `tl.layers.binary._compute_threshold()` (PR #658)
- All references to `tf.logging` replaced by `tl.logging` (PR #661)
- Duplicated code removed when bias was used (PR #667)
- `tensorlayer.third_party.roi_pooling.roi_pooling.roi_pooling_ops` is now lazy loaded to prevent systematic error raised (PR #675)
- Documentation not build in RTD due to old version of theme in docs directory fixed (PR #703)
- Tutorial:
  - `tutorial_word2vec_basic.py` saving issue #476 fixed (PR #635)
  - All tutorials tested and errors have been fixed (PR #635)

### Dependencies Update

- Update pytest from 3.5.1 to 3.6.0 (PR #647)
- Update progressbar2 from 3.37.1 to 3.38.0 (PR #651)
- Update scikit-image from 0.13.1 to 0.14.0 (PR #656)
- Update keras from 2.1.6 to 2.2.0 (PR #684)
- Update requests from 2.18.4 to 2.19.0 (PR #695)

### Contributors

- @lgarithm: #563
- @DEKHTIARJonathan: #573 #574 #575 #580 #633 #635 #636 #639 #644 #645 #648 #657 #667 #658 #659 #660 #661 #666 #667 #672 #675 #683 #686 #687 #690 #691 #692 #703
- @2wins: #560 #566 #662
- @One-sixth: #579
- @zsdonghao: #587 #588 #639 #685 #697
- @luomai: #639 #677
- @dengyueyun666: #676

## [1.8.5] - 2018-05-09

### Added

- Github Templates added (by @DEKHTIARJonathan)
  - New issues Template
  - New PR Template
- Travis Deploy Automation on new Tag (by @DEKHTIARJonathan)
  - Deploy to PyPI and create a new version.
  - Deploy to Github Releases and upload the wheel files
- PyUP.io has been added to ensure we are compatible with the latest libraries (by @DEKHTIARJonathan)
- `deconv2d` now handling dilation_rate (by @zsdonghao)
- Documentation unittest added (by @DEKHTIARJonathan)
- `test_layers_core` has been added to ensure that `LayersConfig` is abstract.

### Changed

- All Tests Refactored - Now using unittests and runned with PyTest (by @DEKHTIARJonathan)
- Documentation updated (by @zsdonghao)
- Package Setup Refactored (by @DEKHTIARJonathan)
- Dataset Downlaod now using library progressbar2 (by @DEKHTIARJonathan)
- `deconv2d` function transformed into Class (by @zsdonghao)
- `conv1d` function transformed into Class (by @zsdonghao)
- super resolution functions transformed into Class (by @zsdonghao)
- YAPF coding style improved and enforced (by @DEKHTIARJonathan)

### Fixed

- Backward Compatibility Restored with deprecation warnings (by @DEKHTIARJonathan)
- Tensorflow Deprecation Fix (Issue #498):
  - AverageEmbeddingInputlayer (by @zsdonghao)
  - load_mpii_pose_dataset (by @zsdonghao)
- maxPool2D initializer issue #551 (by @zsdonghao)
- `LayersConfig` class has been enforced as abstract
- Pooling Layer Issue #557 fixed (by @zsdonghao)

### Dependencies Update

- scipy>=1.0,<1.1 => scipy>=1.1,<1.2

### Contributors

@zsdonghao @luomai @DEKHTIARJonathan

[Unreleased]: https://github.com/tensorlayer/tensorlayer/compare/1.10.1...master
[1.10.1]: https://github.com/tensorlayer/tensorlayer/compare/1.10.0...1.10.1
[1.10.0]: https://github.com/tensorlayer/tensorlayer/compare/1.9.1...1.10.0
[1.9.1]: https://github.com/tensorlayer/tensorlayer/compare/1.9.0...1.9.1
[1.9.0]: https://github.com/tensorlayer/tensorlayer/compare/1.8.5...1.9.0
[1.8.5]: https://github.com/tensorlayer/tensorlayer/compare/1.8.4...1.8.5
