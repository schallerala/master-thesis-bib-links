ALL_PDF := $(shell ls *.pdf)
ALL_TXT := $(ALL_PDF:.pdf=.txt)


LIST_OF_HAND_DL := sensors-22-03601-v2

ALL_ARXIV_IDS := $(shell egrep -o 'arxiv.org/(abs|pdf)/\d+\.\d+(:?v\d)?' README.md | egrep -o '\d+\.\d+(:?v\d)?$$')
GIT_REPOS := $(shell egrep -oi '[^\.]github.com/([\w\./-~\-]+){2}' README.md | egrep -io 'github.com/[^/]+/[^/]+' | sed 's/github\.com\///')



print-git:
	echo $(GIT_REPOS) | tr ' ' '\n'

clone-git:
	@# example: https://github.com/Ki6an/fastT5.git
	echo $(GIT_REPOS) | tr ' ' '\n' | awk '{ print "https://github.com/" $$1 ".git" }' | xargs -I{} sh -c 'GIT=$$(echo "{}"); FOLDER=$$(basename  -s .git "$$GIT"); (test -d $$FOLDER && echo $$FOLDER already cloned) || (git clone "$$GIT" && echo "/$$FOLDER" >> ./.gitignore)'
	make ignore-git-clone


print-paper:
	echo $(ALL_ARXIV_IDS) | tr ' ' '\n'

dl-paper:
	echo $(ALL_ARXIV_IDS) | tr ' ' '\n' | awk '{ print "https://arxiv.org/pdf/" $$1 ".pdf" }' | xargs wget -N --user-agent Master
	make ignore-pdf-txt


.gitignore:
	touch .gitignore
	echo ".vscode" >> .gitignore

ignore-pdf-txt: .gitignore
	echo $(ALL_ARXIV_IDS) $(LIST_OF_HAND_DL) | tr ' ' '\n' | awk '{ print "/" $$1 }' | xargs -I{} sh -c 'grep -q {} .gitignore || (echo {}.pdf >> .gitignore && echo {}.txt >> .gitignore)'


ignore-git-clone: .gitignore
	echo $(GIT_REPOS) | tr ' ' '\n' | xargs -L 1 basename | awk '{ print "/" $$1 }' | xargs -I{} sh -c 'grep -q {} .gitignore || echo {} >> .gitignore'


%.txt: %.pdf
	pdftotext $< $@


all2txt: $(ALL_TXT)


all: clone-git dl-paper all2txt
	cd temporal_localization && make all


.PHONY: all clone-git dl-paper all2txt print-git print-paper ignore-pdf-txt ignore-git-clone