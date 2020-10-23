// code by jph
package ch.ethz.idsc.subare.ch05.racetrack;

import java.io.File;

import ch.ethz.idsc.tensor.ext.HomeDirectory;
import junit.framework.TestCase;

public class VI_RaceTrackTest extends TestCase {
  public void testSimple() throws Exception {
    File file = HomeDirectory.Pictures(getClass().getSimpleName() + ".gif");
    assertFalse(file.exists());
    VI_RaceTrack.make("track2", 4, file);
    assertTrue(file.exists());
    file.delete();
  }
}
