// code by jph
package ch.ethz.idsc.subare.core;

import java.io.File;

public enum Settings {
  ;
  private static final File ROOT = new File("/home/datahaki");

  public static File root() {
    return ROOT;
  }
}
