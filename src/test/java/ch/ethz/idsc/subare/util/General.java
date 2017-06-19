package ch.ethz.idsc.subare.util;

import junit.framework.TestCase;

class Test {
  public Test() {
    String callerClassName = new Exception().getStackTrace()[2].getClassName();
    System.out.println(callerClassName);
  }
}

public class General extends TestCase {
  public void testSimple() {
    new Test();
  }
}
