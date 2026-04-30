MAKEFLAGS += --no-builtin-rules
MAKEFLAGS += --no-builtin-variables
SHELL := /usr/bin/dash


apptainer.sif: export TMPDIR = /home/jannis/s/a/phd/nongeneric_free_fermions/
apptainer.sif: dockerfile apptainer.def
	docker build --network=host -t nongeneric_free_fermions .
	apptainer build apptainer.sif apptainer.def


srs := rsync -avh --info=progress2 -e "ssh -t -c aes128-ctr -o compression=no -x"
remote := uts_hpc:s/a/phd/nongeneric_free_fermions

send:
	$(srs) --filter=':- .gitignore' -R \
	apptainer.sif hpc_run.bash src \
	$(remote)

receive:
	$(srs) $(remote)/output/* output

