// code by jph
package ch.ethz.idsc.subare.core;

import java.io.File;

public enum Settings {
  ;
  private static final File HOME = new File("/home/datahaki");

  public static File home() {
    return HOME;
  }
}
