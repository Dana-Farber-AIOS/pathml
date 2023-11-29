@echo off

rem xmlindent.bat: a batch file for prettifying blocks of XML

rem Required JARs: bioformats_package.jar

setlocal
set BF_DIR=%~dp0
if "%BF_DIR:~-1%" == "\" set BF_DIR=%BF_DIR:~0,-1%

set BF_PROG=loci.formats.tools.XMLIndent
call "%BF_DIR%\bf.bat" %*
