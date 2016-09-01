## install java
1. download java source from oracle
2. unpacking the source code
3. add environment value to /etc/profile 

    export JAVA_HOME=/opt/jdk1.8.0_71
    export JRE_HOME=/opt/jdk1.8.0_71/jre
    export CLASSPATH=.:$CLASSPATH:$JAVA_HOME/lib:$JRE_HOME/lib
    export PATH=$PATH:$JAVA_HOME/bin:$JRE_HOME/bin

4. checking the java system environment 
    java -version


## install docker
I have download a book called "Docker Cooking Book"