# Add a CHANGELOG entry for app changes
if !git.modified_files.include?("CHANGELOG.md")
    fail("Please include a CHANGELOG entry. \nYou can find it at [CHANGELOG.md](https://github.com/tensorlayer/tensorlayer/blob/master/CHANGELOG.md).")
    message "Note, we hard-wrap at 80 chars and use 2 spaces after the last line."
end

# Add ability to modify PR rather than just add comments
# https://github.com/danger/danger/issues/825#issuecomment-303691442
github.api.update_pull_request(
    github.pr_json.base.repo.full_name,
    github.pr_json.number,
    {:body => github.pr_body}
)
