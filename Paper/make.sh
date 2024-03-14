#!/bin/bash
#cp paper.tex ./tex/paper.tex;
#cp mendeley.bib ./tex/mendeley.bib;
#cd tex;
pdflatex -interaction=nonstopmode paper;
bibtex paper;
pdflatex -interaction=nonstopmode paper;
pdflatex -interaction=nonstopmode paper;
#rm paper.tex
#rm mendeley.bib
#mv paper.pdf ../paper.pdf

rm paper.aux;
rm paper.bbl;
rm paper.blg;
rm paper.log;