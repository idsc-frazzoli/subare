// code by jph
package ch.ethz.idsc.subare.util;

public class GlobalAssert {
  public static void of(boolean myBoolean) {
    assert myBoolean;
    if (!myBoolean)
      new Exception().printStackTrace();
  }
}
