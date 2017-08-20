// code by jph
package ch.ethz.idsc.subare.util;

public enum GlobalAssert {
  ;
  public static void that(boolean myBoolean) {
    assert myBoolean;
    if (!myBoolean)
      throw new RuntimeException();
  }
}
