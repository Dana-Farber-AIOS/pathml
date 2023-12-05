@echo off

rem domainlist.bat: a batch file for listing supported domains in Bio-Formats

rem Required JARs: bioformats_package.jar

setlocal
set BF_DIR=%~dp0
if "%BF_DIR:~-1%" == "\" set BF_DIR=%BF_DIR:~0,-1%

set BF_PROG=loci.formats.tools.PrintDomains
call "%BF_DIR%\bf.bat" %*
