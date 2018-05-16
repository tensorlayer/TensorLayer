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
- Tutorials:
  - `tutorial_tfslim` has been introduced to show how to use `SlimNetsLayer` (by @2wins in #560).
- Test:
  - `Layer_DeformableConvolution_Test` added to reproduce issue #572 with deformable convolution (by @DEKHTIARJonathan in #573)
- CI Tool:
  - Danger CI has been added to enforce the update of the changelog (by @lgarithm and @DEKHTIARJonathan in #563)
  - https://github.com/apps/stale/ added to clean stale issues (by @DEKHTIARJonathan in #573)
- Layer:
  - ElementwiseLambdaLayer added to use custom function to connect multiple layer inputs. (by @One-sixth in #579)

### Changed
- Tensorflow CPU & GPU dependencies moved to separated requirement files in order to allow PyUP.io to parse them (by @DEKHTIARJonathan in #573)

### Deprecated

### Removed

### Fixed
- Issue #498 - Deprecation Warning Fix in `tl.layers.RNNLayer` with `inspect` (by @DEKHTIARJonathan in #574)
- Issue #498 - Deprecation Warning Fix in `tl.files` with truth value of an empty array is ambiguous (by @DEKHTIARJonathan in #575)
- Issue #572 with deformable convolution fixed (by @DEKHTIARJonathan in #573)

### Security

### Dependencies Update

### Contributors
@lgarithm @DEKHTIARJonathan @2wins @One-sixth


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

[Unreleased]: https://github.com/tensorlayer/tensorlayer/compare/1.8.5...master
[1.8.5]: https://github.com/tensorlayer/tensorlayer/compare/1.8.4...1.8.5
