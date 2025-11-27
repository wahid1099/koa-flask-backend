# Implementation Plan

- [ ] 1. Set up CSS theme system with custom properties

  - Create CSS variables for light and dark themes (colors, backgrounds, text)
  - Define theme-specific styles using data-theme attribute selector
  - Ensure smooth transitions between theme changes
  - _Requirements: 1.3, 2.4, 3.3_

- [ ] 2. Implement theme toggle functionality

  - [ ] 2.1 Create theme toggle button in HTML with icon/text indicator

    - Add button element to the UI in a visible location
    - Include visual indicators (moon/sun icons or text)
    - _Requirements: 1.1, 3.1_

  - [ ] 2.2 Implement JavaScript theme manager functions

    - Write `initTheme()` to load and apply saved theme on page load
    - Write `toggleTheme()` to switch between dark and light modes
    - Write `applyTheme(themeName)` to update DOM with theme
    - Write `saveTheme(themeName)` to persist to localStorage
    - _Requirements: 1.2, 1.4, 1.5_

  - [ ] 2.3 Write property test for theme toggle state switching

    - **Property 1: Theme toggle switches state**
    - **Validates: Requirements 1.2**

  - [ ] 2.4 Write property test for theme persistence round trip

    - **Property 3: Theme persistence round trip**
    - **Validates: Requirements 1.4, 1.5**

  - [ ] 2.5 Write unit tests for theme toggle edge cases
    - Test default theme when localStorage is empty
    - Test invalid theme values default to light theme
    - Test theme toggle with both initial states
    - _Requirements: 1.2, 1.4, 1.5_

- [ ] 3. Implement developer profile section

  - [ ] 3.1 Add developer photo to static directory

    - Create or add developer photo file to `/static` directory
    - Use appropriate image format (jpg, png, webp)
    - _Requirements: 2.2_

  - [ ] 3.2 Create developer profile HTML structure

    - Add profile section container to HTML
    - Add image element for developer photo with alt text
    - Add text elements for developer name and optional details
    - Position profile section appropriately (header or footer)
    - _Requirements: 2.1, 2.2, 2.3_

  - [ ] 3.3 Style developer profile section for both themes

    - Add CSS styles for profile section layout
    - Ensure profile styling adapts to light and dark themes
    - Make profile section responsive for different screen sizes
    - _Requirements: 2.4_

  - [ ] 3.4 Write property test for profile section theme matching

    - **Property 4: Developer profile styling matches theme**
    - **Validates: Requirements 2.4**

  - [ ] 3.5 Write unit tests for developer profile rendering
    - Test that profile section exists in DOM
    - Test that developer photo has proper alt text
    - Test that developer name is displayed
    - _Requirements: 2.1, 2.2, 2.3_

- [ ] 4. Integrate theme system with existing UI elements

  - [ ] 4.1 Update existing HTML elements to use CSS variables

    - Modify form elements (input, button) to use theme variables
    - Update result display area to use theme variables
    - Update heading and text elements to use theme variables
    - _Requirements: 1.3_

  - [ ] 4.2 Write property test for UI element theme consistency
    - **Property 2: Theme changes update all UI elements**
    - **Validates: Requirements 1.3**

- [ ] 5. Add accessibility and polish

  - [ ] 5.1 Implement accessibility features

    - Add ARIA labels to theme toggle button
    - Ensure keyboard navigation works for theme toggle
    - Verify color contrast meets WCAG standards in both themes
    - Add focus indicators that work in both themes
    - _Requirements: 3.1_

  - [ ] 5.2 Add visual feedback and transitions
    - Implement hover states for theme toggle button
    - Add smooth CSS transitions for theme changes
    - Test visual feedback across different browsers
    - _Requirements: 3.2, 3.3_

- [ ] 6. Final checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.
