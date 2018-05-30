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

### Deprecated

### Removed

### Fixed

### Security

### Dependencies Update

### Contributors

** DO NOT MODIFY THIS SECTION ! **

======================= END: TEMPLATE TO KEEP IN CASE OF NEED ===================

-->

<!-- YOU CAN EDIT FROM HERE -->

## [Unreleased]

### Added
- API:
  - `tl.alphas` and `tl.alphas_like` added following the tf.ones/zeros and tf.zeros_like/ones_like (by @DEKHTIARJonathan in #580)
- CI Tool:
  - [Stale Probot](https://github.com/probot/stale) added to clean stale issues (by @DEKHTIARJonathan in #573)
  - [Changelog Probot](https://github.com/mikz/probot-changelog) Configuration added (by @DEKHTIARJonathan in #637)
  - Travis Builds now handling a matrix of TF Version from TF==1.6.0 to TF==1.8.0 (by @DEKHTIARJonathan in #644)
  - CircleCI added to build and upload Docker Containers for each PR merged and tag release (by @DEKHTIARJonathan in #648)
- Docker:
  - Containers for each release and for each PR merged on master built (by @DEKHTIARJonathan in #648)
  - Containers built in the following configurations: py2+cpu, py2+gpu, py3+cpu, and py3+gpu (by @DEKHTIARJonathan in #648)
- Documentation:
  - Release semantic version added on index page (by @DEKHTIARJonathan in #633)
  - Optimizers page added (by @DEKHTIARJonathan in #636)
  - `AMSGrad` added on Optimizers page added (by @DEKHTIARJonathan in #636)
- Layer:
  - ElementwiseLambdaLayer added to use custom function to connect multiple layer inputs (by @One-sixth in #579)
- Optimizer:
  - AMSGrad Optimizer added based on `On the Convergence of Adam and Beyond (ICLR 2018)` (by @DEKHTIARJonathan in #636)
- Test:
  - `test_utils_predict.py` added to reproduce and fix issue #288 (by @2wins in #566)
  - `Layer_DeformableConvolution_Test` added to reproduce issue #572 with deformable convolution (by @DEKHTIARJonathan in #573)
  - `Array_Op_Alphas_Test` and `Array_Op_Alphas_Like_Test` added to test `tensorlayer/array_ops.py` file (by @DEKHTIARJonathan in #580)
  - `test_optimizer_amsgrad.py` added to test `AMSGrad` optimizer (by @DEKHTIARJonathan in #636)
  - `test_logging.py` added to insure robustness of the logging API (by @DEKHTIARJonathan in #645)
- Tutorials:
  - `tutorial_tfslim` has been introduced to show how to use `SlimNetsLayer` (by @2wins in #560).

### Changed
- Tensorflow CPU & GPU dependencies moved to separated requirement files in order to allow PyUP.io to parse them (by @DEKHTIARJonathan in #573)
- The document of LambdaLayer for linking it with ElementwiseLambdaLayer (by @zsdonghao in #587)
- RTD links point to stable documentation instead of latest used for development (by @DEKHTIARJonathan in #633)
- TF Version older than 1.6.0 are officially unsupported and raises an exception (by @DEKHTIARJonathan in #644)
- Readme Badges Updated with Support Python and Tensorflow Versions (by @DEKHTIARJonathan in #644)
- TL logging API has been consistent with TF logging API and thread-safe (by @DEKHTIARJonathan in #645)
- Relative Imports changed for absolute imports (by @DEKHTIARJonathan in #657)
- `tl.files` refactored into a directory with numerous files (by @DEKHTIARJonathan in #657)
- `tl.files.voc_dataset` fixed because of original Pascal VOC website was down (by @DEKHTIARJonathan in #657)
- extra requirements hidden inside the library added in the project requirements (by @DEKHTIARJonathan in #657)
- requirements files refactored in `requirements/` directory (by @DEKHTIARJonathan in #657) 
- README.md and other markdown files have been refactored and cleaned. (by @zsdonghao @DEKHTIARJonathan @luomai in #639)
- Ternary Convolution Layer added in unittest (by @DEKHTIARJonathan in #658)
- Convolution Layers unittests have been cleaned & refactored (by @DEKHTIARJonathan in #658)

### Deprecated

### Removed

### Fixed
- Issue #498 - Deprecation Warning Fix in `tl.layers.RNNLayer` with `inspect` (by @DEKHTIARJonathan in #574)
- Issue #498 - Deprecation Warning Fix in `tl.files` with truth value of an empty array is ambiguous (by @DEKHTIARJonathan in #575)
- Issue #565 related to `tl.utils.predict` fixed - `np.hstack` problem in which the results for multiple batches are stacked along `axis=1` (by @2wins in #566)
- Issue #572 with `tl.layers.DeformableConv2d` fixed (by @DEKHTIARJonathan in #573)
- Typo of the document of ElementwiseLambdaLayer (by @zsdonghao in #588)
- Error in `tl.layers.TernaryConv2d` fixed - self.inputs not defined (by @DEKHTIARJonathan in #658)
- Deprecation warning fixed in `tl.layers.binary._compute_threshold()` (by @DEKHTIARJonathan in #658)

### Security

### Dependencies Update
- Update pytest from 3.5.1 to 3.6.0 (by @DEKHTIARJonathan and @pyup-bot in #647)
- Update progressbar2 from 3.37.1 to 3.38.0 (by @DEKHTIARJonathan and @pyup-bot in #651)
- Update scikit-image from 0.13.1 to 0.14.0 (by @DEKHTIARJonathan and @pyup-bot in #656)

### Contributors
@lgarithm @DEKHTIARJonathan @2wins @One-sixth @zsdonghao @luomai

## [1.8.6] - 2018-05-30

### Added
- API:
  - `tl.alphas` and `tl.alphas_like` added following the tf.ones/zeros and tf.zeros_like/ones_like (by @DEKHTIARJonathan in #580)
- CI Tool:
  - [Stale Probot](https://github.com/probot/stale) added to clean stale issues (by @DEKHTIARJonathan in #573)
  - [Changelog Probot](https://github.com/mikz/probot-changelog) Configuration added (by @DEKHTIARJonathan in #637)
  - Travis Builds now handling a matrix of TF Version from TF==1.6.0 to TF==1.8.0 (by @DEKHTIARJonathan in #644)
  - CircleCI added to build and upload Docker Containers for each PR merged and tag release (by @DEKHTIARJonathan in #648)
- Docker:
  - Containers for each release and for each PR merged on master built (by @DEKHTIARJonathan in #648)
  - Containers built in the following configurations: py2+cpu, py2+gpu, py3+cpu, and py3+gpu (by @DEKHTIARJonathan in #648)
- Documentation:
  - Release semantic version added on index page (by @DEKHTIARJonathan in #633)
  - Optimizers page added (by @DEKHTIARJonathan in #636)
  - `AMSGrad` added on Optimizers page added (by @DEKHTIARJonathan in #636)
- Layer:
  - ElementwiseLambdaLayer added to use custom function to connect multiple layer inputs (by @One-sixth in #579)
- Optimizer:
  - AMSGrad Optimizer added based on `On the Convergence of Adam and Beyond (ICLR 2018)` (by @DEKHTIARJonathan in #636)
- Test:
  - `test_utils_predict.py` added to reproduce and fix issue #288 (by @2wins in #566)
  - `Layer_DeformableConvolution_Test` added to reproduce issue #572 with deformable convolution (by @DEKHTIARJonathan in #573)
  - `Array_Op_Alphas_Test` and `Array_Op_Alphas_Like_Test` added to test `tensorlayer/array_ops.py` file (by @DEKHTIARJonathan in #580)
  - `test_optimizer_amsgrad.py` added to test `AMSGrad` optimizer (by @DEKHTIARJonathan in #636)
  - `test_logging.py` added to insure robustness of the logging API (by @DEKHTIARJonathan in #645)
- Tutorials:
  - `tutorial_tfslim` has been introduced to show how to use `SlimNetsLayer` (by @2wins in #560).

### Changed
- Tensorflow CPU & GPU dependencies moved to separated requirement files in order to allow PyUP.io to parse them (by @DEKHTIARJonathan in #573)
- The document of LambdaLayer for linking it with ElementwiseLambdaLayer (by @zsdonghao in #587)
- RTD links point to stable documentation instead of latest used for development (by @DEKHTIARJonathan in #633)
- TF Version older than 1.6.0 are officially unsupported and raises an exception (by @DEKHTIARJonathan in #644)
- Readme Badges Updated with Support Python and Tensorflow Versions (by @DEKHTIARJonathan in #644)
- TL logging API has been consistent with TF logging API and thread-safe (by @DEKHTIARJonathan in #645)
- Relative Imports changed for absolute imports (by @DEKHTIARJonathan in #657)
- `tl.files` refactored into a directory with numerous files (by @DEKHTIARJonathan in #657)
- `tl.files.voc_dataset` fixed because of original Pascal VOC website was down (by @DEKHTIARJonathan in #657)
- extra requirements hidden inside the library added in the project requirements (by @DEKHTIARJonathan in #657)
- requirements files refactored in `requirements/` directory (by @DEKHTIARJonathan in #657)
- README.md and other markdown files have been refactored and cleaned. (by @zsdonghao @DEKHTIARJonathan @luomai in #639)
- Ternary Convolution Layer added in unittest (by @DEKHTIARJonathan in #658)
- Convolution Layers unittests have been cleaned & refactored (by @DEKHTIARJonathan in #658)

### Fixed
- Issue #498 - Deprecation Warning Fix in `tl.layers.RNNLayer` with `inspect` (by @DEKHTIARJonathan in #574)
- Issue #498 - Deprecation Warning Fix in `tl.files` with truth value of an empty array is ambiguous (by @DEKHTIARJonathan in #575)
- Issue #565 related to `tl.utils.predict` fixed - `np.hstack` problem in which the results for multiple batches are stacked along `axis=1` (by @2wins in #566)
- Issue #572 with `tl.layers.DeformableConv2d` fixed (by @DEKHTIARJonathan in #573)
- Typo of the document of ElementwiseLambdaLayer (by @zsdonghao in #588)

### Dependencies Update
- Update pytest from 3.5.1 to 3.6.0 (by @DEKHTIARJonathan and @pyup-bot in #647)
- Update progressbar2 from 3.37.1 to 3.38.0 (by @DEKHTIARJonathan and @pyup-bot in #651)
- Update scikit-image from 0.13.1 to 0.14.0 (by @DEKHTIARJonathan and @pyup-bot in #656)

### Contributors
@lgarithm @DEKHTIARJonathan @2wins @One-sixth @zsdonghao @luomai

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

[Unreleased]: https://github.com/tensorlayer/tensorlayer/compare/1.8.6rc0...master
[1.8.6]: https://github.com/tensorlayer/tensorlayer/compare/1.8.6rc0...1.8.5
[1.8.5]: https://github.com/tensorlayer/tensorlayer/compare/1.8.4...1.8.5
