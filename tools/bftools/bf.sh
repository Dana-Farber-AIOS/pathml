#!/usr/bin/env bash

# bf.sh: the script that actually launches a command line tool

BF_DIR=`dirname "$0"`

# Include the master configuration file.
source "$BF_DIR/config.sh"

# Check that a command to run was specified.
if [ -z "$BF_PROG" ]
then
  echo The command to launch must be set in the BF_PROG environment variable.
  exit 1
fi

# Prepare the flags.
if [ -z "$BF_FLAGS" ]
then
   BF_FLAGS=""
fi

# Set the max heap size.
if [ -z "$BF_MAX_MEM" ]
then
  # Set a reasonable default max heap size.
  BF_MAX_MEM="512m"
fi
BF_FLAGS="$BF_FLAGS -Xmx$BF_MAX_MEM"

# Skip the update check if the NO_UPDATE_CHECK flag is set.
if [ -n "$NO_UPDATE_CHECK" ]
then
  BF_FLAGS="$BF_FLAGS -Dbioformats_can_do_upgrade_check=false"
fi

# Run profiling if the BF_PROFILE flag is set.
if [ -n "$BF_PROFILE" ]
then
  # Set default profiling depth
  if [ -z "$BF_PROFILE_DEPTH" ]
  then
    BF_PROFILE_DEPTH="30"
  fi
  BF_FLAGS="$BF_FLAGS -agentlib:hprof=cpu=samples,depth=$BF_PROFILE_DEPTH,file=$BF_PROG.hprof"
fi

# Use any available proxy settings.
BF_FLAGS="$BF_FLAGS -Dhttp.proxyHost=$PROXY_HOST -Dhttp.proxyPort=$PROXY_PORT"

# Run the command!
if [ -n "$BF_DEVEL" ]
then
  # Developer environment variable set; launch with existing classpath.
  java $BF_FLAGS $BF_PROG "$@"
else
  # Developer environment variable unset; add JAR libraries to classpath.
  if [ -e "$BF_JAR_DIR/bioformats_package.jar" ]
  then
    BF_CP="$BF_JAR_DIR/bioformats_package.jar:$BF_CP"
  elif [ -e "$BF_JAR_DIR/formats-gpl.jar" ]
  then
    BF_CP="$BF_JAR_DIR/formats-gpl.jar:$BF_JAR_DIR/bio-formats-tools.jar:$BF_CP"
  else
    # Libraries not found; issue an error.
    echo "Required JAR libraries not found. Please download:"
    echo "  bioformats_package.jar"
    echo "from:"
    echo "  https://downloads.openmicroscopy.org/latest/bio-formats/artifacts/"
    echo "and place in the same directory as the command line tools."
    exit 2
  fi
  if [ -e "$BF_JAR_DIR/bio-formats-testing-framework.jar" ]
  then
    BF_CP="$BF_CP:$BF_JAR_DIR/bio-formats-testing-framework.jar"
  fi
  java $BF_FLAGS -cp "$BF_DIR:$BF_CP" $BF_PROG "$@"
fi
