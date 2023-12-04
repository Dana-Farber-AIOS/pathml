@echo off

rem tiffcomment.bat: a batch file for extracting the comment
rem                  (OME-XML block or otherwise) from a TIFF file

rem Required JARs: bioformats_package.jar

setlocal
set BF_DIR=%~dp0
if "%BF_DIR:~-1%" == "\" set BF_DIR=%BF_DIR:~0,-1%

set BF_PROG=loci.formats.tools.TiffComment
call "%BF_DIR%\bf.bat" %*
