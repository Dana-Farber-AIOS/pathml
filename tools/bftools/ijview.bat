@echo off

rem ijview.bat: a batch file for displaying an image file in ImageJ
rem             using the Bio-Formats Importer plugin

rem Required JARs: bioformats_package.jar, ij.jar

setlocal
set BF_DIR=%~dp0
if "%BF_DIR:~-1%" == "\" set BF_DIR=%BF_DIR:~0,-1%

call "%BF_DIR%\config.bat"

if "%BF_DEVEL%" == "" (
  rem Developer environment variable unset; add JAR libraries to classpath.
  if exist "%BF_JAR_DIR%\ij.jar" (
    set BF_CP="%BF_JAR_DIR%\ij.jar"
  ) else (
    rem Libraries not found; issue an error.
    echo Required JAR libraries not found. Please download:
    echo   ij.jar
    echo from:
    echo   https://imagej.nih.gov/ij/upgrade/
    echo and place in the same directory as the command line tools.
    goto end
  )
  if exist "%BF_JAR_DIR%\bio-formats_plugins.jar" (
    set BF_CP=%BF_CP%;"%BF_JAR_DIR%\bio-formats_plugins.jar"
  ) else if not exist "%BF_JAR_DIR%\bioformats_package.jar" (
    rem Libraries not found; issue an error.
    echo Required JAR libraries not found. Please download:
    echo   bioformats_package.jar
    echo from:
    echo   https://www.openmicroscopy.org/bio-formats/downloads
    echo and place in the same directory as the command line tools.
    goto end
  )
)

set BF_PROG=loci.plugins.in.Importer
call "%BF_DIR%\bf.bat" %*

:end
