// code by jph
package ch.ethz.idsc.subare.util;

import java.util.Optional;

import junit.framework.TestCase;

public class JavaTest extends TestCase {
  public void testOptional() {
    assertTrue(Optional.ofNullable(3).isPresent());
    assertFalse(Optional.ofNullable(null).isPresent());
  }
}
