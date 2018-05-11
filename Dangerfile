# Add a CHANGELOG entry for app changes
if !git.modified_files.include?("CHANGELOG.md")
    fail("Please include a CHANGELOG entry. \nYou can find it at [CHANGELOG.md](https://github.com/tensorlayer/tensorlayer/blob/master/CHANGELOG.md).")
    message "Note, we hard-wrap at 80 chars and use 2 spaces after the last line."
end

# Look for prose issues
prose.lint_files markdown_files

# Look for spelling issues
prose.ignored_words = ["orta", "artsy", "cocoapods"]
prose.check_spelling markdown_files
