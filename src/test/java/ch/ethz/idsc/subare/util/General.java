package ch.ethz.idsc.subare.util;

import junit.framework.TestCase;

public class General extends TestCase {
  static class Test {
    public Test() {
      String callerClassName = new Exception().getStackTrace()[2].getClassName();
      System.out.println(callerClassName);
    }
  }

  public void testSimple() {
    new Test();
  }
}
