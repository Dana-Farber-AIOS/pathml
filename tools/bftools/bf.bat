@echo off

rem bf.bat: the batch file that actually launches a command line tool

setlocal enabledelayedexpansion
set BF_DIR=%~dp0
if "%BF_DIR:~-1%" == "\" set BF_DIR=%BF_DIR:~0,-1%

rem Include the master configuration file.
call "%BF_DIR%\config.bat"

rem Check that a command to run was specified.
if "%BF_PROG%" == "" (
  echo The command to launch must be set in the BF_PROG environment variable.
  goto end
)

rem Set the max heap size.
if "%BF_MAX_MEM%" == "" (
  rem Set a reasonable default max heap size.
  set BF_MAX_MEM=512m
)
set BF_FLAGS=%BF_FLAGS% -Xmx%BF_MAX_MEM%

rem Skip the update check if the NO_UPDATE_CHECK flag is set.
if not "%NO_UPDATE_CHECK%" == "" (
  set BF_FLAGS=%BF_FLAGS% -Dbioformats_can_do_upgrade_check=false
)

rem Run profiling if the BF_PROFILE flag is set.
if not "%BF_PROFILE%" == "" (
  if "%BF_PROFILE_DEPTH%" == "" (
    rem Set default profiling depth
    set BF_PROFILE_DEPTH=30
  )
  set BF_FLAGS=%BF_FLAGS% -agentlib:hprof=cpu=samples,depth=!BF_PROFILE_DEPTH!,file=%BF_PROG%.hprof
)


rem Use any available proxy settings.
set BF_FLAGS=%BF_FLAGS% -Dhttp.proxyHost=%PROXY_HOST% -Dhttp.proxyPort=%PROXY_PORT%

rem Run the command!
if not "%BF_DEVEL%" == "" (
  rem Developer environment variable set; launch with existing classpath.
  java %BF_FLAGS% %BF_PROG% %*
  goto end
)

rem Developer environment variable unset; add JAR libraries to classpath.
if exist "%BF_JAR_DIR%\bioformats_package.jar" (
    set BF_CP=%BF_CP%;"%BF_JAR_DIR%\bioformats_package.jar"
) else if exist "%BF_JAR_DIR%\formats-gpl.jar" (
    set BF_CP=%BF_CP%;"%BF_JAR_DIR%\formats-gpl.jar";"%BF_JAR_DIR%\bio-formats-tools.jar"
) else (
  rem Libraries not found; issue an error.
  echo Required JAR libraries not found. Please download:
  echo   bioformats_package.jar
  echo from:
  echo   https://downloads.openmicroscopy.org/latest/bio-formats/artifacts/
  echo and place in the same directory as the command line tools.
  goto end
)
if exist "%BF_JAR_DIR/bio-formats-testing-framework.jar" (
  set BF_CP=%BF_CP%;"%BF_JAR_DIR%\bio-formats-testing-framework.jar"
)

java %BF_FLAGS% -cp "%BF_DIR%";%BF_CP% %BF_PROG% %*

:end
