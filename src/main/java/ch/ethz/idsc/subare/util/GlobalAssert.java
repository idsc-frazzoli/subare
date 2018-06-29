// code by jph
package ch.ethz.idsc.subare.util;

public enum GlobalAssert {
  ;
  public static void that(boolean status) {
    assert status;
    if (!status)
      throw new RuntimeException();
  }
}
