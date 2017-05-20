// code by jph
package ch.ethz.idsc.subare.util;

import java.io.File;

public enum UserHome {
  ;
  public static File file(String filename) {
    return new File(System.getProperty("user.home"), filename);
  }
}
